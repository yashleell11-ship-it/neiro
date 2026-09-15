"""One OpenAI-compatible streaming client, used for both tiers.

Promoting to the 3090 Ti is a `base_url` change and nothing else — that
is the whole reason this speaks the OpenAI shape rather than ollama's
native `/api/chat`. llama-server, ollama, and vLLM all expose
`/v1/chat/completions`, so the client code never forks per backend.

Two things this handles that a naive SSE reader gets wrong:

  - **Tool-call fragments.** Arguments stream in as partial JSON strings
    across many deltas, indexed by position. They have to be accumulated
    per index and only parsed once the stream finishes — parsing early
    gets you a JSONDecodeError on `{"na`. The accumulator exists from
    Stage 0 and stays inert until Stage 3 adds tools.

  - **Thinking mode, and the way it actually fails here.** Every current
    Qwen3.5/3.6 checkpoint reasons by default and will emit tens of
    thousands of reasoning tokens before the first speakable word —
    which on a voice assistant is the entire latency budget spent before
    she says anything.

    Measured against a real server on 2026-09-13 (ollama 0.33.2 serving
    Qwen3.5-4B-Q4_K_M), and it is worse than the plan assumed. On
    `/v1/chat/completions` the reasoning does **not** arrive as a
    `<think>` tag in the content at all: it goes into a separate
    `delta.reasoning` field while `delta.content` stays `""` for the
    whole stream. So a client that reads only `content` — and a guard
    that looks for `"<think"` in the text — sees a perfectly well-formed
    stream that produces no words, burns the entire `max_tokens` budget,
    and raises nothing. Silent failure with no error, which is the exact
    class of bug this project exists to design out.

    Worse, on that endpoint **neither** documented switch works:
    `think: false` and `chat_template_kwargs.enable_thinking: false`
    were both accepted and both ignored. Only ollama's native
    `/api/chat` with `think: false` actually suppressed it.

    So `check_not_thinking` takes the whole delta, not just the text,
    and fails on three separate signatures: a `<think` opener in the
    content, a populated reasoning field, and a stream that produced
    reasoning but no content at all.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from elizabeth.config import Elizabeth
from elizabeth.state import Locality, ToolCall

# How many leading characters to inspect for a `<think` opener. Enough
# to catch it past any leading whitespace, small enough that it can't
# swallow a real reply.
THINK_SNIFF_CHARS = 32

# Field names backends use for out-of-band reasoning on an otherwise
# OpenAI-shaped delta. ollama 0.33 uses "reasoning"; vLLM and several
# OpenAI-compatible proxies use "reasoning_content". Both mean the same
# thing: tokens are being spent somewhere the caller cannot see.
REASONING_FIELDS = ("reasoning", "reasoning_content")

_DIAGNOSIS = (
    "Verified on this machine 2026-09-13: ollama's /v1/chat/completions ignores "
    "BOTH `think: false` and `chat_template_kwargs.enable_thinking: false`, and "
    "hides the reasoning in `delta.reasoning` while `delta.content` stays empty. "
    "Only ollama's native /api/chat with `think: false` suppressed it. For "
    "llama-server use `--reasoning-budget 0 --reasoning-format none`. Whatever "
    "the backend, verify with curl before trusting it."
)


class ThinkingModeError(RuntimeError):
    """Raised when a model emits a reasoning block despite being told not
    to. Loud on purpose — silently stripping it would hide a
    misconfiguration that costs the entire latency budget.
    """


@dataclass
class StreamAccumulator:
    """Collects a streamed response: text, tool calls, finish reason."""

    text: str = ""
    finish_reason: str | None = None
    reasoning_chars: int = 0  # tokens spent where the caller cannot see them
    _tool_fragments: dict[int, dict] = field(default_factory=dict, repr=False)

    def add_delta(self, delta: dict) -> str:
        """Apply one `choices[0].delta`; return newly added text.

        Reasoning is counted, never concatenated into `text` — it must
        not reach the chunker or the TTS, but "how much was spent
        invisibly" is the number that diagnoses a silent stall.
        """
        added = delta.get("content") or ""
        if added:
            self.text += added

        for key in REASONING_FIELDS:
            hidden = delta.get(key)
            if hidden:
                self.reasoning_chars += len(hidden)

        for fragment in delta.get("tool_calls") or []:
            index = fragment.get("index", 0)
            slot = self._tool_fragments.setdefault(index, {"name": "", "arguments": ""})
            function = fragment.get("function") or {}
            if function.get("name"):
                slot["name"] = function["name"]
            if function.get("arguments"):
                # Arguments arrive as partial JSON text — concatenate,
                # never parse until the stream is done.
                slot["arguments"] += function["arguments"]

        return added

    def tool_calls(self) -> list[ToolCall]:
        """Parse accumulated fragments into ToolCalls. Only valid once
        the stream has finished.
        """
        calls = []
        for index in sorted(self._tool_fragments):
            slot = self._tool_fragments[index]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                # A truncated or malformed tool call is dropped rather
                # than guessed at — Stage 3's registry validates every
                # argument anyway, and half-parsed args are exactly the
                # kind of thing that should never reach it.
                continue
            calls.append(ToolCall(name=slot["name"], args=args, index=index))
        return calls


def parse_sse_line(line: str) -> dict | None:
    """Parse one Server-Sent Events line into a chunk dict.

    Returns None for keepalives, comments, blank lines, and the
    terminal `[DONE]` sentinel.
    """
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def check_delta_not_thinking(delta: dict) -> None:
    """Raise if a single delta carries out-of-band reasoning.

    This is the check that catches the real-world failure: the content
    field stays empty and legal-looking while the budget drains into
    `delta.reasoning`. Checking the text alone never fires here.
    """
    for key in REASONING_FIELDS:
        if delta.get(key):
            raise ThinkingModeError(
                f"The model is streaming reasoning in `delta.{key}` while "
                "`delta.content` stays empty — the entire token budget is being "
                f"spent before she says a word, with no error. {_DIAGNOSIS}"
            )


def check_not_thinking(text: str) -> None:
    """Raise if the model opened with an inline reasoning block.

    Backends that keep reasoning *in* the content (llama-server with
    `--reasoning-format none` misconfigured, most vLLM setups) fail this
    way instead.
    """
    if "<think" in text[:THINK_SNIFF_CHARS].lower():
        raise ThinkingModeError(
            "The model emitted a <think> block in its content despite thinking "
            f"mode being disabled. {_DIAGNOSIS}"
        )


def check_produced_words(accumulator: StreamAccumulator) -> None:
    """Raise if a finished stream produced reasoning but no speakable text.

    The last line of defence: some backend, some day, will hide
    reasoning under a field name not in `REASONING_FIELDS`. A turn that
    consumed a budget and yielded nothing to say is that bug, whatever
    it is called.
    """
    if not accumulator.text.strip() and accumulator.reasoning_chars:
        raise ThinkingModeError(
            f"The stream finished with no speakable text at all, after "
            f"{accumulator.reasoning_chars} characters of hidden reasoning. "
            f"{_DIAGNOSIS}"
        )


class OpenAiCompatLlm:
    """protocols.LLM implementation. TIERABLE — the only difference
    between the laptop and the 3090 Ti is `base_url`.
    """

    locality = Locality.TIERABLE

    def __init__(self, cfg: Elizabeth | None = None, base_url: str | None = None) -> None:
        self._cfg = cfg or Elizabeth()
        self._base_url = (base_url or self._cfg.llm.base_url).rstrip("/")

    def build_request(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """The request body. Separated out so it's testable without a
        server, and so the thinking-mode flags are visible in one place.
        """
        body: dict = {
            "model": self._cfg.llm.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._cfg.llm.max_tokens,
            "temperature": self._cfg.llm.temperature,
            # Belt and braces — different backends honour different ones,
            # and the llama.cpp issues above mean none can be trusted alone.
            "think": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        """Yield `{"text": str}` for speakable fragments and a final
        `{"done": StreamAccumulator}` when the stream ends.
        """
        accumulator = StreamAccumulator()
        checked_thinking = False

        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client,
            client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=self.build_request(messages, tools),
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]

                delta = choice.get("delta") or {}
                # Before anything else: this fires on the FIRST reasoning
                # delta, so a misconfigured backend costs one chunk of
                # latency rather than a whole max_tokens budget.
                check_delta_not_thinking(delta)

                added = accumulator.add_delta(delta)

                if not checked_thinking and len(accumulator.text) >= THINK_SNIFF_CHARS:
                    check_not_thinking(accumulator.text)
                    checked_thinking = True

                if added:
                    yield {"text": added}

                if choice.get("finish_reason"):
                    accumulator.finish_reason = choice["finish_reason"]

        # A short reply may never reach THINK_SNIFF_CHARS — check once more.
        if not checked_thinking:
            check_not_thinking(accumulator.text)
        check_produced_words(accumulator)

        yield {"done": accumulator}
