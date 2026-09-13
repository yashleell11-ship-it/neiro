"""The ollama-native client.

It exists because ollama's OpenAI-compatible endpoint cannot turn
thinking off — measured, not assumed. These tests pin the shape
differences that make it a separate client, and the one behaviour it
must share with the OpenAI path: the orchestrator cannot tell them
apart.
"""

from __future__ import annotations

import json

import pytest

from neiro.llm.ollama_native import (
    THINKING_FIELD,
    OllamaNativeLlm,
    check_chunk_not_thinking,
    parse_ndjson_line,
)
from neiro.llm.openai_compat import OpenAiCompatLlm, ThinkingModeError
from neiro.state import Locality


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

    def test_tool_calls_are_reshaped_into_the_shared_accumulator_form(self) -> None:
        # ollama sends whole tool calls; the OpenAI path sends indexed
        # fragments. Reshaping here means one accumulator, not two.
        from neiro.llm.openai_compat import StreamAccumulator

        chunk = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "set_volume", "arguments": {"percent": 30}}}],
            }
        }
        message = chunk["message"]
        delta = {
            "content": "",
            "tool_calls": [
                {
                    "index": i,
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": json.dumps(c["function"]["arguments"]),
                    },
                }
                for i, c in enumerate(message["tool_calls"])
            ],
        }
        acc = StreamAccumulator()
        acc.add_delta(delta)
        calls = acc.tool_calls()
        assert len(calls) == 1
        assert calls[0].name == "set_volume"
        assert calls[0].args == {"percent": 30}
