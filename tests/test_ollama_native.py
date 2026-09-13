"""The ollama-native client.

It exists because ollama's OpenAI-compatible endpoint cannot turn
thinking off — measured, not assumed. These tests pin the shape
differences that make it a separate client, and the one behaviour it
must share with the OpenAI path: the orchestrator cannot tell them
apart.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from neiro.llm.ollama_native import (
    THINKING_FIELD,
    OllamaNativeLlm,
    check_chunk_not_thinking,
    parse_ndjson_line,
)
from neiro.llm.openai_compat import OpenAiCompatLlm, StreamAccumulator, ThinkingModeError
from neiro.state import Locality

TOOLS = [{"type": "function", "function": {"name": "set_volume", "parameters": {}}}]


def _tool_message(name: str, arguments: dict) -> dict:
    """One NDJSON object as ollama emits a whole tool call: arguments
    are a JSON object, not the OpenAI path's partial string."""
    return {
        "model": "neiro-4b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        },
        "done": False,
    }


END = {"model": "neiro-4b", "message": {"role": "assistant", "content": ""}, "done": True}


def _serving(lines: list[dict]) -> OllamaNativeLlm:
    """A client whose /api/chat replies with the given NDJSON stream."""
    body = "".join(json.dumps(line) + "\n" for line in lines).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, content=body)

    return OllamaNativeLlm(base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def _drain(llm: OllamaNativeLlm) -> StreamAccumulator:
    async def go() -> StreamAccumulator:
        final = None
        async for event in llm.stream([{"role": "user", "content": "hi"}], TOOLS):
            if "done" in event:
                final = event["done"]
        assert final is not None
        return final

    return asyncio.run(go())


class TestNdjson:
    def test_parses_a_real_line(self) -> None:
        # Verbatim from ollama 0.33.2.
        line = '{"model":"neiro-4b","message":{"role":"assistant","content":"Hello"},"done":false}'
        chunk = parse_ndjson_line(line)
        assert chunk is not None
        assert chunk["message"]["content"] == "Hello"

    @pytest.mark.parametrize("line", ["", "   ", "not json", "[1,2,3]", "null"])
    def test_junk_is_skipped_not_raised(self, line: str) -> None:
        assert parse_ndjson_line(line) is None

    def test_there_is_no_data_prefix_or_done_sentinel(self) -> None:
        # SSE habits break here: ollama sends bare NDJSON, and the end is
        # a "done": true field rather than a [DONE] line.
        assert parse_ndjson_line("data: {}") is None
        assert parse_ndjson_line("[DONE]") is None
        assert parse_ndjson_line('{"done":true}') == {"done": True}


class TestThinkingGuard:
    def test_reasoning_in_the_thinking_field_is_caught(self) -> None:
        with pytest.raises(ThinkingModeError, match="message.thinking"):
            check_chunk_not_thinking({"message": {"content": "", THINKING_FIELD: "Let me"}})

    def test_a_normal_chunk_passes(self) -> None:
        check_chunk_not_thinking({"message": {"content": "Hey"}})
        check_chunk_not_thinking({})
        check_chunk_not_thinking({"message": {"content": "", THINKING_FIELD: ""}})

    def test_the_request_sets_the_flag_that_actually_works(self) -> None:
        # think:false on /api/chat is the ONLY one measured to work.
        # /v1's think and chat_template_kwargs were both ignored.
        body = OllamaNativeLlm().build_request([{"role": "user", "content": "hi"}])
        assert body["think"] is False

    def test_it_uses_ollamas_option_names(self) -> None:
        # num_predict, not max_tokens — the OpenAI name is silently
        # ignored here, which would mean unbounded replies.
        body = OllamaNativeLlm().build_request([])
        assert "num_predict" in body["options"]
        assert "max_tokens" not in body


class TestInterchangeable:
    def test_both_clients_present_the_same_surface(self) -> None:
        # The orchestrator must not be able to tell them apart — that is
        # what writing protocols.py first bought.
        for attribute in ("stream", "build_request", "locality"):
            assert hasattr(OllamaNativeLlm, attribute)
            assert hasattr(OpenAiCompatLlm, attribute)

    def test_both_are_tierable(self) -> None:
        assert OllamaNativeLlm.locality is Locality.TIERABLE
        assert OpenAiCompatLlm.locality is Locality.TIERABLE


class TestToolCalls:
    """ollama sends whole tool calls; the OpenAI path sends indexed
    fragments. The client reshapes into the shared accumulator so there
    is one accumulator, not two — and these drive the real `stream()`,
    because a re-implementation of the reshaping inside a test is what
    let the index bug below through.
    """

    def test_a_whole_call_is_reshaped_into_the_shared_accumulator_form(self) -> None:
        acc = _drain(_serving([_tool_message("set_volume", {"percent": 30}), END]))
        calls = acc.tool_calls()
        assert len(calls) == 1
        assert calls[0].name == "set_volume"
        assert calls[0].args == {"percent": 30}
        assert acc.finish_reason == "stop"

    def test_calls_in_separate_messages_keep_distinct_indices(self) -> None:
        # The accumulator keys by index for the WHOLE stream and
        # concatenates arguments into the slot. A per-message index
        # restarts at 0, so two calls in two messages both land in slot
        # 0 as '{"percent":30}{"app":"firefox"}' — unparseable, dropped
        # with no error. Both must survive, JSON intact.
        acc = _drain(
            _serving(
                [
                    _tool_message("set_volume", {"percent": 30}),
                    _tool_message("open_app", {"app": "firefox"}),
                    END,
                ]
            )
        )
        calls = acc.tool_calls()
        assert [(c.name, c.args) for c in calls] == [
            ("set_volume", {"percent": 30}),
            ("open_app", {"app": "firefox"}),
        ]
        assert [c.index for c in calls] == [0, 1]

    def test_two_calls_in_one_message_are_still_distinct(self) -> None:
        # The other way the counter could go wrong: one slot per message.
        message = _tool_message("set_volume", {"percent": 30})
        message["message"]["tool_calls"].append(
            {"function": {"name": "open_app", "arguments": {"app": "firefox"}}}
        )
        acc = _drain(_serving([message, END]))
        assert [c.index for c in acc.tool_calls()] == [0, 1]

    def test_the_request_carries_the_tools(self) -> None:
        body = OllamaNativeLlm().build_request([], TOOLS)
        assert body["tools"] == TOOLS
