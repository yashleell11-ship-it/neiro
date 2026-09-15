"""Tests for the emotion-tag stream parser.

This is the single channel carrying Elizabeth's emotional state to her
voice, her face, and the next turn's prompt — the project's entire
differentiator runs through these ~40 lines. The failure modes that
matter are subtle: a tag split across stream chunks that never resolves,
or a missing tag that swallows her words instead of just her expression.
"""

from __future__ import annotations

import pytest

from elizabeth.llm.emotion_tag import FALLBACK_AFTER_CHARS, EmotionTagParser
from elizabeth.state import EmotionLabel


def feed_all(parser: EmotionTagParser, chunks: list[str]) -> tuple[list, str]:
    """Helper: feed chunks, collect published states and forwarded text."""
    states = []
    text = ""
    for chunk in chunks:
        state, forwarded = parser.feed(chunk)
        if state is not None:
            states.append(state)
        text += forwarded
    state, forwarded = parser.flush()
    if state is not None:
        states.append(state)
    text += forwarded
    return states, text


class TestWholeTagInOneChunk:
    def test_tag_is_parsed_and_stripped(self) -> None:
        parser = EmotionTagParser()
        state, text = parser.feed("<e:happy:8> Hey, you sound tired.")
        assert state is not None
        assert state.label is EmotionLabel.HAPPY
        assert state.intensity == pytest.approx(8 / 9)
        assert text == "Hey, you sound tired."

    def test_every_valid_label_parses(self) -> None:
        for label in EmotionLabel:
            parser = EmotionTagParser()
            state, text = parser.feed(f"<e:{label.value}:5> hello")
            assert state is not None, f"{label.value} failed to parse"
            assert state.label is label
            assert text == "hello"

    def test_intensity_digit_maps_to_zero_to_one(self) -> None:
        parser = EmotionTagParser()
        state, _ = parser.feed("<e:sad:0> x")
        assert state.intensity == pytest.approx(0.0)

        parser = EmotionTagParser()
        state, _ = parser.feed("<e:sad:9> x")
        assert state.intensity == pytest.approx(1.0)

    def test_valence_and_arousal_come_along(self) -> None:
        # The face wants floats to blend, not a bare categorical.
        parser = EmotionTagParser()
        state, _ = parser.feed("<e:happy:9> x")
        assert state.valence > 0
        assert state.arousal > 0


class TestTagSplitAcrossChunks:
    def test_tag_arriving_one_character_at_a_time(self) -> None:
        # The realistic case: tokens stream in tiny fragments.
        parser = EmotionTagParser()
        states, text = feed_all(parser, list("<e:angry:7> Fine."))
        assert len(states) == 1
        assert states[0].label is EmotionLabel.ANGRY
        assert text == "Fine."

    def test_nothing_is_forwarded_before_the_tag_resolves(self) -> None:
        parser = EmotionTagParser()
        state, text = parser.feed("<e:ha")
        assert state is None
        assert text == ""  # must not leak a partial tag to the TTS

    def test_tag_split_at_awkward_boundaries(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:re", "laxed", ":3", "> all", " good"])
        assert len(states) == 1
        assert states[0].label is EmotionLabel.RELAXED
        assert text == "all good"


class TestFallback:
    def test_no_tag_falls_back_to_neutral_and_keeps_the_words(self) -> None:
        # The words matter more than the expression — losing her reply
        # would be far worse than losing her face.
        long_reply = "x" * (FALLBACK_AFTER_CHARS + 10)
        parser = EmotionTagParser()
        states, text = feed_all(parser, [long_reply])
        assert len(states) == 1
        assert states[0].label is EmotionLabel.NEUTRAL
        assert text == long_reply

    def test_short_untagged_reply_is_not_swallowed(self) -> None:
        # "Sure." is shorter than the fallback threshold, so only flush()
        # can rescue it. Without flush handling this would vanish.
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["Sure."])
        assert len(states) == 1
        assert states[0].label is EmotionLabel.NEUTRAL
        assert text == "Sure."

    def test_malformed_tag_is_treated_as_no_tag(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:ecstatic:5> hello there friend"])
        assert states[0].label is EmotionLabel.NEUTRAL
        assert "ecstatic" in text  # forwarded verbatim, not silently eaten

    def test_missing_intensity_digit_is_treated_as_no_tag(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:happy> hello there my friend ok"])
        assert states[0].label is EmotionLabel.NEUTRAL
        assert "<e:happy>" in text

    def test_two_digit_intensity_is_treated_as_no_tag(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:happy:10> hello there my friend"])
        assert states[0].label is EmotionLabel.NEUTRAL
        assert "hello there my friend" in text  # her words survive regardless


class TestPassThroughAfterResolution:
    def test_later_chunks_pass_straight_through(self) -> None:
        parser = EmotionTagParser()
        parser.feed("<e:happy:8> Hey")
        state, text = parser.feed(" there")
        assert state is None
        assert text == " there"

    def test_state_is_published_exactly_once(self) -> None:
        parser = EmotionTagParser()
        states, _ = feed_all(parser, ["<e:sad:2> one", " two", " three"])
        assert len(states) == 1

    def test_a_later_tag_like_string_is_not_reinterpreted(self) -> None:
        # A stray "<e:angry:9>" mid-reply must not hijack her face.
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:happy:8> hi <e:angry:9> there"])
        assert len(states) == 1
        assert states[0].label is EmotionLabel.HAPPY
        assert "<e:angry:9>" in text

    def test_resolved_flag(self) -> None:
        parser = EmotionTagParser()
        assert not parser.resolved
        parser.feed("<e:happy:8> hi")
        assert parser.resolved


class TestEdgeCases:
    def test_empty_stream_resolves_to_neutral_with_no_text(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, [])
        assert len(states) == 1
        assert states[0].label is EmotionLabel.NEUTRAL
        assert text == ""

    def test_empty_chunks_are_harmless(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["", "<e:happy:8>", "", " hi", ""])
        assert len(states) == 1
        assert text == "hi"

    def test_tag_only_reply_forwards_nothing(self) -> None:
        parser = EmotionTagParser()
        states, text = feed_all(parser, ["<e:neutral:5>"])
        assert len(states) == 1
        assert text == ""

    def test_case_insensitive_label(self) -> None:
        # The model will occasionally capitalise. Accept it rather than
        # losing the expression to a cosmetic difference.
        parser = EmotionTagParser()
        state, text = parser.feed("<e:Happy:8> hi")
        assert state.label is EmotionLabel.HAPPY
        assert text == "hi"

    def test_flush_after_resolution_is_a_noop(self) -> None:
        parser = EmotionTagParser()
        parser.feed("<e:happy:8> hi")
        state, text = parser.flush()
        assert state is None
        assert text == ""
