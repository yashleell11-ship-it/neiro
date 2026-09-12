"""Tests for the OpenAI-compatible LLM client.

Everything here runs on synthetic stream data — no model, no server. The
parts worth pinning down are the ones a naive SSE reader gets wrong:
tool-call arguments arriving as partial JSON across many deltas, and a
reasoning block slipping through despite being disabled.
"""

from __future__ import annotations

import json

import pytest

from neiro.llm.openai_compat import (
    StreamAccumulator,
    ThinkingModeError,
    check_not_thinking,
    parse_sse_line,
)


class TestParseSseLine:
    def test_parses_a_data_line(self) -> None:
        assert parse_sse_line('data: {"a": 1}') == {"a": 1}

    def test_done_sentinel_is_not_a_chunk(self) -> None:
        assert parse_sse_line("data: [DONE]") is None

    def test_blank_and_comment_lines_are_skipped(self) -> None:
        assert parse_sse_line("") is None
        assert parse_sse_line("   ") is None
        assert parse_sse_line(": keepalive") is None

    def test_non_data_lines_are_skipped(self) -> None:
        assert parse_sse_line("event: message") is None

    def test_malformed_json_does_not_raise(self) -> None:
        # A truncated line mid-stream must not kill the turn.
        assert parse_sse_line('data: {"broken') is None

    def test_tolerates_missing_space_after_colon(self) -> None:
        assert parse_sse_line('data:{"a": 1}') == {"a": 1}


class TestStreamAccumulatorText:
    def test_accumulates_text(self) -> None:
        acc = StreamAccumulator()
        acc.add_delta({"content": "Hello"})
        acc.add_delta({"content": " there"})
        assert acc.text == "Hello there"

    def test_returns_only_the_newly_added_text(self) -> None:
        acc = StreamAccumulator()
        assert acc.add_delta({"content": "one"}) == "one"
        assert acc.add_delta({"content": "two"}) == "two"

    def test_empty_and_null_content_are_harmless(self) -> None:
        acc = StreamAccumulator()
        assert acc.add_delta({}) == ""
        assert acc.add_delta({"content": None}) == ""
        assert acc.text == ""


class TestStreamAccumulatorToolCalls:
    def test_arguments_arriving_as_partial_json_are_reassembled(self) -> None:
        # THE case that matters: arguments stream as fragments of a JSON
        # string. Parsing any one fragment alone raises JSONDecodeError.
        acc = StreamAccumulator()
        acc.add_delta(
            {"tool_calls": [{"index": 0, "function": {"name": "set_volume", "arguments": ""}}]}
        )
        for fragment in ['{"per', 'cent"', ": 4", "0}"]:
            acc.add_delta({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]})

        calls = acc.tool_calls()
        assert len(calls) == 1
        assert calls[0].name == "set_volume"
        assert calls[0].args == {"percent": 40}

    def test_parallel_tool_calls_are_kept_separate_by_index(self) -> None:
        acc = StreamAccumulator()
        acc.add_delta({"tool_calls": [{"index": 0, "function": {"name": "a", "arguments": "{}"}}]})
        acc.add_delta({"tool_calls": [{"index": 1, "function": {"name": "b", "arguments": "{}"}}]})
        calls = acc.tool_calls()
        assert [c.name for c in calls] == ["a", "b"]
        assert [c.index for c in calls] == [0, 1]

    def test_malformed_arguments_are_dropped_not_guessed(self) -> None:
        # Stage 3's registry validates every argument, and half-parsed
        # args are exactly what must never reach it.
        acc = StreamAccumulator()
        acc.add_delta(
            {"tool_calls": [{"index": 0, "function": {"name": "x", "arguments": '{"a": '}}]}
        )
        assert acc.tool_calls() == []

    def test_empty_arguments_become_an_empty_dict(self) -> None:
        acc = StreamAccumulator()
        acc.add_delta({"tool_calls": [{"index": 0, "function": {"name": "ping", "arguments": ""}}]})
        calls = acc.tool_calls()
        assert len(calls) == 1
        assert calls[0].args == {}

    def test_a_fragment_with_no_name_is_ignored(self) -> None:
        acc = StreamAccumulator()
        acc.add_delta({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]})
        assert acc.tool_calls() == []

    def test_no_tool_calls_is_the_normal_case(self) -> None:
        # The accumulator exists from Stage 0 and stays inert until
        # Stage 3 — a plain chat turn must produce nothing.
        acc = StreamAccumulator()
        acc.add_delta({"content": "just talking"})
        assert acc.tool_calls() == []


class TestThinkingModeDetection:
    def test_plain_reply_passes(self) -> None:
        check_not_thinking("Hey, you sound tired.")  # must not raise

    def test_think_block_raises_loudly(self) -> None:
        with pytest.raises(ThinkingModeError, match="latency budget"):
            check_not_thinking("<think>The user is asking about...")

    def test_leading_whitespace_does_not_hide_it(self) -> None:
        with pytest.raises(ThinkingModeError):
            check_not_thinking("\n  <think>hmm")

    def test_case_insensitive(self) -> None:
        with pytest.raises(ThinkingModeError):
            check_not_thinking("<THINK>")

    def test_the_word_think_in_normal_speech_is_fine(self) -> None:
        # "I think so" must not trip it — only the tag form does.
        check_not_thinking("I think so, yeah.")

    def test_a_think_tag_far_into_the_reply_is_not_sniffed(self) -> None:
        # Only the opening is inspected; this is deliberate, so a long
        # normal reply that happens to contain the string can't trip it.
        check_not_thinking("x" * 100 + "<think>")


class TestBuildRequest:
    def _client(self):
        from neiro.config import Neiro
        from neiro.llm.openai_compat import OpenAiCompatLlm

        return OpenAiCompatLlm(Neiro())

    def test_thinking_is_disabled_both_ways(self) -> None:
        body = self._client().build_request([{"role": "user", "content": "hi"}])
        assert body["think"] is False
        assert body["chat_template_kwargs"]["enable_thinking"] is False

    def test_streams_by_default(self) -> None:
        body = self._client().build_request([{"role": "user", "content": "hi"}])
        assert body["stream"] is True

    def test_tools_are_omitted_when_not_supplied(self) -> None:
        body = self._client().build_request([{"role": "user", "content": "hi"}])
        assert "tools" not in body

    def test_tools_enable_auto_tool_choice(self) -> None:
        tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
        body = self._client().build_request([{"role": "user", "content": "hi"}], tools=tools)
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"

    def test_body_is_json_serialisable(self) -> None:
        body = self._client().build_request([{"role": "user", "content": "hi"}])
        json.dumps(body)  # must not raise
