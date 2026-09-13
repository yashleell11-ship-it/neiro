"""An ollama-native client, because its OpenAI-compatible endpoint cannot
turn thinking off.

This exists for one measured reason (2026-09-13, ollama 0.33.2 serving
Qwen3.5-4B-Q4_K_M). On `/v1/chat/completions`:

    think: false                                  -> ignored
    chat_template_kwargs.enable_thinking: false   -> ignored
    delta.content stays ""; delta.reasoning fills

...so a 60-token request returned no words and spent the whole budget
invisibly. On ollama's own `/api/chat`, `think: false` works: the same
prompt returned "Hello there, friend." with no thinking field at all.

**Why not just use llama-server.** It is the target for both tiers and
its `--reasoning-budget 0` is the real answer — but installing it needs
`pacman`, and this machine has no passwordless sudo. Rather than leave
the whole loop untestable until that happens, this speaks ollama's
native protocol behind the same `protocols.LLM` interface. The
orchestrator cannot tell them apart, which is the entire point of having
written the Protocol first.

The shapes differ in ways that matter:

  - **NDJSON, not SSE.** One JSON object per line, no `data:` prefix and
    no `[DONE]` sentinel; the final object carries `"done": true`.
  - **`message.content`, not `choices[0].delta.content`.**
  - **Reasoning lands in `message.thinking`**, so the same
    out-of-band-reasoning guard applies and is reused rather than
    reimplemented.
  - **`options.num_predict`, not `max_tokens`.**
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from neiro.config import Neiro
from neiro.llm.openai_compat import (
    StreamAccumulator,
    ThinkingModeError,
    check_not_thinking,
    check_produced_words,
)
from neiro.state import Locality

# ollama's own spelling for out-of-band reasoning on /api/chat.
THINKING_FIELD = "thinking"


def parse_ndjson_line(line: str) -> dict | None:
    """One line of ollama's stream. None for blanks and unparseable text."""
    line = line.strip()
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def check_chunk_not_thinking(chunk: dict) -> None:
    """Raise if ollama is streaming reasoning rather than words.

    Same failure as the OpenAI path, different field name. Fires on the
    first such chunk so a misconfigured server costs one chunk of latency
    rather than a whole `num_predict` budget.
    """
    if (chunk.get("message") or {}).get(THINKING_FIELD):
        raise ThinkingModeError(
            "ollama is streaming into `message.thinking` — the request did not "
            "disable reasoning. Set `think: false` on /api/chat (it is ignored on "
            "/v1/chat/completions; verified 2026-09-13)."
        )


class OllamaNativeLlm:
    """protocols.LLM over ollama's `/api/chat`. TIERABLE like the other."""

    locality = Locality.TIERABLE

    def __init__(
        self,
        cfg: Neiro | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cfg = cfg or Neiro()
        self._base_url = (base_url or self._cfg.llm.base_url).rstrip("/")
        # The seam that lets a test drive `stream()` over a recorded
        # NDJSON fixture with no server. A re-implementation of the
        # reshaping inside a test proves nothing about the client.
        self._transport = transport

    def build_request(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body: dict = {
            "model": self._cfg.llm.model,
            "messages": messages,
            "stream": True,
            # The one that actually works. Not belt-and-braces here: the
            # other spellings were measured to be ignored, and listing
            # them would imply a redundancy that does not exist.
            "think": False,
            "options": {
                "num_predict": self._cfg.llm.max_tokens,
                "temperature": self._cfg.llm.temperature,
            },
        }
        if tools:
            body["tools"] = tools
        return body

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        """Yield `{"text": str}` per speakable fragment, then
        `{"done": StreamAccumulator}` — identical to the OpenAI client,
        so the orchestrator has one shape to handle.
        """
        accumulator = StreamAccumulator()
        checked = False
        # The accumulator keys tool calls by index for the WHOLE stream
        # and concatenates arguments into the slot — that is how the
        # OpenAI shape streams one call's arguments across many deltas.
        # ollama's position is per message, so calls that arrive in
        # separate messages must not both be index 0: they would share
        # a slot, the arguments would become two JSON objects glued
        # together, and tool_calls() would drop both without a word.
        next_tool_index = 0

        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=5.0), transport=self._transport
            ) as client,
            client.stream(
                "POST", f"{self._base_url}/api/chat", json=self.build_request(messages, tools)
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = parse_ndjson_line(line)
                if chunk is None:
                    continue
                check_chunk_not_thinking(chunk)

                message = chunk.get("message") or {}
                # ollama sends whole tool calls, not the OpenAI fragment
                # stream, so they are reshaped into the accumulator's
                # indexed form rather than given a second code path.
                delta: dict = {"content": message.get("content") or ""}
                if message.get("tool_calls"):
                    calls = message["tool_calls"]
                    delta["tool_calls"] = [
                        {
                            "index": next_tool_index + i,
                            "function": {
                                "name": (call.get("function") or {}).get("name", ""),
                                "arguments": json.dumps(
                                    (call.get("function") or {}).get("arguments", {})
                                ),
                            },
                        }
                        for i, call in enumerate(calls)
                    ]
                    next_tool_index += len(calls)

                added = accumulator.add_delta(delta)
                if not checked and len(accumulator.text) >= 32:
                    check_not_thinking(accumulator.text)
                    checked = True
                if added:
                    yield {"text": added}
                if chunk.get("done"):
                    accumulator.finish_reason = chunk.get("done_reason") or "stop"

        if not checked:
            check_not_thinking(accumulator.text)
        check_produced_words(accumulator)
        yield {"done": accumulator}
