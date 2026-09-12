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

  - **Thinking mode.** Every current Qwen3.5/3.6 checkpoint reasons by
    default and will emit a `<think>` block of up to tens of thousands
    of tokens before the first speakable word — which on a voice
    assistant is the entire latency budget spent before she says
    anything. Two llama.cpp issues report the documented disable switch
    being silently ignored, both closed without a fix, so this asserts
    rather than trusts: `strip_thinking` drops any turn whose opening
    characters contain `<think` and says so loudly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from neiro.config import Neiro
from neiro.state import Locality, ToolCall

# How many leading characters to inspect for a `<think` opener. Enough
# to catch it past any leading whitespace, small enough that it can't
# swallow a real reply.
THINK_SNIFF_CHARS = 32


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
    _tool_fragments: dict[int, dict] = field(default_factory=dict, repr=False)

    def add_delta(self, delta: dict) -> str:
        """Apply one `choices[0].delta`; return newly added text."""
        added = delta.get("content") or ""
        if added:
            self.text += added

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


def check_not_thinking(text: str) -> None:
    """Raise if the model opened with a reasoning block."""
    if "<think" in text[:THINK_SNIFF_CHARS].lower():
        raise ThinkingModeError(
            "The model emitted a <think> block despite thinking mode being "
            "disabled. This costs the entire latency budget before she says a "
            "word. Check the request sets think=false (ollama) or "
            "--reasoning-budget 0 --reasoning-format none (llama-server), and "
            "verify with curl — two llama.cpp issues report the documented "
            "switch being silently ignored."
        )


class OpenAiCompatLlm:
    """protocols.LLM implementation. TIERABLE — the only difference
    between the laptop and the 3090 Ti is `base_url`.
    """

    locality = Locality.TIERABLE

    def __init__(self, cfg: Neiro | None = None, base_url: str | None = None) -> None:
        self._cfg = cfg or Neiro()
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

                added = accumulator.add_delta(choice.get("delta") or {})

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

        yield {"done": accumulator}
