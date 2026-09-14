"""The rules the persona/Hinglish/tool-calling LoRA's data path is built
on. `training/recipes/persona_train.py` needs CUDA torch and cannot be
imported here, which is exactly why everything worth asserting lives in
`neiro.training.llm_data` instead — importable and testable in the
runtime venv, no GPU required.

The load-bearing test here is `TestTrainingChatTemplate`: an earlier
draft of the masking scheme used the "obvious" incremental-prefix trick
(render `messages[:1]`, `messages[:2]`, ... and diff token counts) and it
silently misaligned every label after the first assistant turn, because
Qwen3.5's chat template renders an assistant turn differently depending
on whether it is the LAST one — a property that changes when the message
list is truncated. `build_training_chat_template` fixes this with
transformers' own `{% generation %}` mechanism instead; these tests use
tiny synthetic templates (never the real 8 GB checkpoint) to pin the
masking behaviour and the vendor-template-shape assertion.

    /home/yash/code/neiro/.venv/bin/python -m pytest tests/test_persona_train.py
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from neiro.evals.latency import percentile
from neiro.training.llm_data import (
    IGNORE_INDEX,
    SkipLog,
    ToolCallCheck,
    apply_assistant_mask,
    build_training_chat_template,
    check_tool_call_shape,
    choose_max_length,
    filter_publishable,
    find_tool_call_turn,
    gguf_convert_commands,
    iter_jsonl,
    parse_tool_call_arguments,
    shuffle_stream,
)

# --------------------------------------------------------------------------
# reading the prepared JSONL


class TestIterJsonl:
    def test_one_record_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n')
        assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text('{"a": 1}\n\n  \n{"a": 2}\n')
        assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]

    def test_an_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text("")
        assert list(iter_jsonl(path)) == []


# --------------------------------------------------------------------------
# streaming shuffle


class TestShuffleStream:
    def test_output_is_a_permutation_of_the_input(self) -> None:
        items = list(range(200))
        out = list(shuffle_stream(items, buffer_size=20, seed=0))
        assert sorted(out) == items

    def test_the_same_seed_gives_the_same_order(self) -> None:
        items = list(range(200))
        first = list(shuffle_stream(items, buffer_size=20, seed=7))
        second = list(shuffle_stream(items, buffer_size=20, seed=7))
        assert first == second

    def test_different_seeds_give_different_orders(self) -> None:
        items = list(range(200))
        first = list(shuffle_stream(items, buffer_size=20, seed=1))
        second = list(shuffle_stream(items, buffer_size=20, seed=2))
        assert first != second

    def test_a_buffer_of_one_is_not_an_error(self) -> None:
        # No real shuffling is possible with a size-1 buffer, but it must
        # still yield every item exactly once.
        items = list(range(10))
        assert sorted(shuffle_stream(items, buffer_size=1, seed=0)) == items

    def test_a_shuffle_actually_reorders_a_long_sequence(self) -> None:
        items = list(range(500))
        out = list(shuffle_stream(items, buffer_size=100, seed=0))
        assert out != items

    def test_buffer_size_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="buffer_size"):
            list(shuffle_stream([1, 2, 3], buffer_size=0, seed=0))

    def test_fewer_items_than_the_buffer_still_yields_everything(self) -> None:
        items = list(range(5))
        assert sorted(shuffle_stream(items, buffer_size=100, seed=0)) == items


# --------------------------------------------------------------------------
# tool-call argument parsing


class TestParseToolCallArguments:
    def test_a_json_string_is_parsed_into_a_mapping(self) -> None:
        messages = [
            {"role": "user", "content": "convert"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"amount": 1000}'},
                    }
                ],
            },
        ]
        out = parse_tool_call_arguments(messages)
        assert out is not None
        assert out[1]["tool_calls"][0]["function"]["arguments"] == {"amount": 1000}

    def test_an_already_parsed_mapping_passes_through(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}],
            }
        ]
        out = parse_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}

    def test_an_empty_arguments_string_becomes_an_empty_mapping(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": ""}}],
            }
        ]
        out = parse_tool_call_arguments(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {}

    def test_malformed_json_returns_none_and_is_counted(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}],
            }
        ]
        log = SkipLog()
        assert parse_tool_call_arguments(messages, log) is None
        assert log.counts["bad_tool_args"] == 1

    def test_messages_without_tool_calls_are_untouched(self) -> None:
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert parse_tool_call_arguments(messages) == messages

    def test_the_input_is_not_mutated(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
            }
        ]
        original = json.loads(json.dumps(messages))
        parse_tool_call_arguments(messages)
        assert messages == original


class TestFindToolCallTurn:
    def test_finds_the_first_assistant_turn_with_a_tool_call(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "sure"},
            {"role": "user", "content": "convert 1000"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f"}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ]
        assert find_tool_call_turn(messages) == 3

    def test_no_tool_call_anywhere_is_none(self) -> None:
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert find_tool_call_turn(messages) is None

    def test_an_assistant_turn_with_an_empty_tool_calls_list_does_not_count(self) -> None:
        messages = [{"role": "assistant", "content": "hi", "tool_calls": []}]
        assert find_tool_call_turn(messages) is None


# --------------------------------------------------------------------------
# publishable filtering


class TestFilterPublishable:
    def test_a_non_publishable_source_is_dropped_and_counted(self) -> None:
        records = [{"source": "good"}, {"source": "bad"}, {"source": "good"}]
        per_source = {
            "good": {"weights_publishable": "yes"},
            "bad": {"weights_publishable": "unclear"},
        }
        log = SkipLog()
        kept = list(filter_publishable(records, per_source, log))
        assert kept == [{"source": "good"}, {"source": "good"}]
        assert log.counts["not_publishable"] == 1

    def test_a_source_missing_from_the_table_is_dropped_not_assumed_ok(self) -> None:
        records = [{"source": "unknown"}]
        log = SkipLog()
        assert list(filter_publishable(records, {}, log)) == []
        assert log.counts["not_publishable"] == 1


# --------------------------------------------------------------------------
# SkipLog


class TestSkipLog:
    def test_a_typo_in_a_skip_reason_is_an_error(self) -> None:
        log = SkipLog()
        with pytest.raises(ValueError, match="unknown skip kind"):
            log.skip("nto_publishable")

    def test_counts_add_up(self) -> None:
        log = SkipLog()
        log.skip("bad_tool_args")
        log.skip("bad_tool_args", 2)
        log.skip("no_assistant_tokens")
        assert log.counts["bad_tool_args"] == 3
        assert log.total == 4
        assert log.as_dict()["by_reason"] == {"bad_tool_args": 3, "no_assistant_tokens": 1}


# --------------------------------------------------------------------------
# label masking


class TestApplyAssistantMask:
    def test_kept_positions_keep_their_token_masked_ones_become_ignore_index(self) -> None:
        assert apply_assistant_mask([1, 2, 3, 4], [0, 0, 1, 1]) == [
            IGNORE_INDEX,
            IGNORE_INDEX,
            3,
            4,
        ]

    def test_all_zero_mask_is_all_ignore_index(self) -> None:
        assert apply_assistant_mask([1, 2, 3], [0, 0, 0]) == [IGNORE_INDEX] * 3

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same render"):
            apply_assistant_mask([1, 2, 3], [1, 1])

    def test_ignore_index_is_minus_100(self) -> None:
        assert IGNORE_INDEX == -100


# --------------------------------------------------------------------------
# the training-only chat template patch (tiny synthetic templates —
# never the real 8 GB checkpoint's tokenizer)


class TestBuildTrainingChatTemplate:
    def test_the_vendor_block_is_replaced_with_generation_tags(self) -> None:
        vendor = (
            '    {%- elif message.role == "assistant" %}\n'
            "        {%- set reasoning_content = '' %}\n"
            "        {%- if message.reasoning_content is string %}\n"
            "            {%- set reasoning_content = message.reasoning_content %}\n"
            "        {%- else %}\n"
            "            {%- if '</think>' in content %}\n"
            "                {%- set reasoning_content = content.split('</think>')[0]"
            ".rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
            "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
            "            {%- endif %}\n"
            "        {%- endif %}\n"
            "        {%- set reasoning_content = reasoning_content|trim %}\n"
            "        {%- if loop.index0 > ns.last_query_index %}\n"
            "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content"
            " + '\\n</think>\\n\\n' + content }}\n"
            "        {%- else %}\n"
            "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
            "        {%- endif %}\n"
            "        {%- if message.tool_calls and message.tool_calls is iterable"
            " and message.tool_calls is not mapping %}\n"
            "            {%- for tool_call in message.tool_calls %}\n"
            "                {{- '</function>\\n</tool_call>' }}\n"
            "            {%- endfor %}\n"
            "        {%- endif %}\n"
            "        {{- '<|im_end|>\\n' }}\n"
            '    {%- elif message.role == "tool" %}\n'
            "        placeholder\n"
        )
        patched = build_training_chat_template(vendor)
        assert "{%- generation %}" in patched
        assert "{%- endgeneration %}" in patched
        # The header/think-scaffold branches are untouched — only what
        # follows them (content onward) moved inside the markers.
        assert "<think>\\n' + reasoning_content" in patched

    def test_a_template_that_has_moved_raises_instead_of_silently_no_opping(self) -> None:
        with pytest.raises(ValueError, match="did not match"):
            build_training_chat_template("this is not the qwen3.5 template at all")


# --------------------------------------------------------------------------
# tool-call structural check


class TestCheckToolCallShape:
    def test_a_well_formed_single_call(self) -> None:
        text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nDelhi\n</parameter>\n</function>\n</tool_call>"
        check = check_tool_call_shape(text)
        assert check.well_formed
        assert check.function_names == ("get_weather",)
        assert check.function_known is None

    def test_plain_text_with_no_call_is_not_well_formed(self) -> None:
        check = check_tool_call_shape("Sure, here's the weather in Delhi: sunny.")
        assert not check.well_formed
        assert check.reason == "no <tool_call> block found"

    def test_an_unclosed_parameter_tag_is_not_well_formed(self) -> None:
        text = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</function>\n</tool_call>"
        check = check_tool_call_shape(text)
        assert not check.well_formed
        assert check.reason == "an unclosed <parameter> tag"

    def test_an_empty_function_name_is_not_well_formed(self) -> None:
        text = "<tool_call>\n<function=>\n</function>\n</tool_call>"
        check = check_tool_call_shape(text)
        assert not check.well_formed

    def test_a_call_to_a_declared_function_is_known(self) -> None:
        text = "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        check = check_tool_call_shape(text, declared_names=["get_weather", "convert_currency"])
        assert check.well_formed and check.function_known is True

    def test_a_call_to_an_undeclared_function_is_not_known(self) -> None:
        text = "<tool_call>\n<function=made_up_function>\n</function>\n</tool_call>"
        check = check_tool_call_shape(text, declared_names=["get_weather"])
        assert check.well_formed and check.function_known is False

    def test_no_declared_names_means_unknown_verdict_not_false(self) -> None:
        # None (unanswered) must never be confused with False (checked
        # and wrong) — see the ToolCallCheck docstring.
        text = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
        assert check_tool_call_shape(text).function_known is None

    def test_multiple_calls_are_all_reported(self) -> None:
        text = (
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=b>\n</function>\n</tool_call>"
        )
        check = check_tool_call_shape(text)
        assert check.well_formed
        assert check.function_names == ("a", "b")

    def test_is_a_frozen_dataclass(self) -> None:
        check = check_tool_call_shape("no call")
        with pytest.raises(FrozenInstanceError):
            check.well_formed = True  # type: ignore[misc]
        assert isinstance(check, ToolCallCheck)


# --------------------------------------------------------------------------
# sequence length


class TestChooseMaxLength:
    def test_rounds_up_to_the_nearest_multiple(self) -> None:
        # p90 of a mostly-small distribution with one large outlier.
        lengths = [10] * 90 + [900] * 10
        value = percentile([float(n) for n in lengths], 0.90)
        result = choose_max_length(lengths, 0.90, round_to=64)
        assert result >= value
        assert result % 64 == 0
        assert result - 64 < value  # the SMALLEST multiple at or above it

    def test_matches_the_repos_one_percentile_function(self) -> None:
        lengths = list(range(1, 101))
        expected = 32 * math.ceil(percentile([float(n) for n in lengths], 0.5) / 32)
        assert choose_max_length(lengths, 0.5, round_to=32) == expected

    def test_zero_examples_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero measured"):
            choose_max_length([], 0.9)

    def test_default_round_to_is_64(self) -> None:
        assert choose_max_length([1000], 0.5) % 64 == 0


# --------------------------------------------------------------------------
# GGUF conversion commands


class TestGgufConvertCommands:
    def test_convert_names_the_merged_model_and_the_f16_output(self) -> None:
        convert, _quantize = gguf_convert_commands(
            "models/merged", "models/out-f16.gguf", "models/out-q4.gguf"
        )
        assert convert[0] == "python"
        assert "convert_hf_to_gguf.py" in convert
        assert convert[convert.index("--outfile") + 1] == "models/out-f16.gguf"
        assert convert[convert.index("--outtype") + 1] == "f16"

    def test_quantize_takes_the_f16_output_and_the_requested_type(self) -> None:
        _convert, quantize = gguf_convert_commands("m", "f16.gguf", "q4.gguf", "Q4_K_M")
        assert quantize == ["llama-quantize", "f16.gguf", "q4.gguf", "Q4_K_M"]

    def test_quant_type_defaults_to_q4_k_m(self) -> None:
        _convert, quantize = gguf_convert_commands("m", "f16.gguf", "out.gguf")
        assert quantize[-1] == "Q4_K_M"
