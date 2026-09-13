"""Her state reaching her voice.

A prompt is code here — Qwen3-TTS's `instruct` field is free-form
English, so these strings are the interface and get pinned like one.

The rule this file mostly protects: the instruction is passed
out-of-band and must never be speakable content. A TTS reading its own
stage direction aloud is a specific, embarrassing failure.
"""

from __future__ import annotations

import pytest

from neiro.emotion.voice import exaggeration_for, instruct_for
from neiro.state import EmotionLabel, NeiroState


def state(label: EmotionLabel, intensity: float = 0.6) -> NeiroState:
    return NeiroState.from_label(label, intensity)


class TestInstruct:
    @pytest.mark.parametrize("label", list(EmotionLabel))
    def test_every_label_produces_an_instruction(self, label: EmotionLabel) -> None:
        # A missing mapping would silently fall back to the default voice
        # and the face would be expressive over a flat reading — the
        # mismatch is exactly what people notice.
        text = instruct_for(state(label))
        assert text and text.endswith(".")

    @pytest.mark.parametrize("label", list(EmotionLabel))
    def test_the_voice_is_recognisably_one_person_across_states(self, label: EmotionLabel) -> None:
        assert "young woman" in instruct_for(state(label))

    def test_intensity_changes_the_instruction(self) -> None:
        weak = instruct_for(state(EmotionLabel.HAPPY, 0.2))
        mid = instruct_for(state(EmotionLabel.HAPPY, 0.6))
        strong = instruct_for(state(EmotionLabel.HAPPY, 0.9))
        assert weak != mid != strong
        assert "slightly" in weak
        assert "clearly" in strong

    def test_it_describes_delivery_rather_than_naming_the_emotion(self) -> None:
        # "Say this angrily" produces a caricature. "Clipped, lower, with
        # an edge" produces something a person might actually say.
        angry = instruct_for(state(EmotionLabel.ANGRY, 0.9)).lower()
        assert "angry" not in angry
        assert "clipped" in angry or "edge" in angry

    def test_anger_is_not_shouting(self) -> None:
        # She lives in a hostel room and talks to one person.
        assert "not shouting" in instruct_for(state(EmotionLabel.ANGRY, 0.9))

    def test_instructions_stay_short(self) -> None:
        # TTFA is part of the measured latency budget, and long
        # instructions dilute rather than intensify.
        for label in EmotionLabel:
            assert len(instruct_for(state(label, 0.9)).split()) < 30

    def test_the_instruction_is_not_speakable_content(self) -> None:
        # It must look like a direction, not like a line of dialogue, so
        # a mis-wired call is obvious rather than subtle.
        for label in EmotionLabel:
            text = instruct_for(state(label))
            assert text.startswith("A young woman")

    def test_neutral_is_still_conversational_not_a_null_instruction(self) -> None:
        # An empty instruct string gets whatever the model's default is,
        # which is not a decision anyone made.
        text = instruct_for(state(EmotionLabel.NEUTRAL))
        assert "conversational" in text


class TestExaggeration:
    """Chatterbox's single dial, derived from the same state."""

    def test_neutral_is_the_floor_not_zero(self) -> None:
        # A completely flat reading sounds synthetic even when the words
        # are right.
        assert exaggeration_for(state(EmotionLabel.NEUTRAL)) == pytest.approx(0.25)

    def test_it_rises_with_intensity(self) -> None:
        low = exaggeration_for(state(EmotionLabel.HAPPY, 0.1))
        high = exaggeration_for(state(EmotionLabel.HAPPY, 0.9))
        assert low < high

    @pytest.mark.parametrize("intensity", [0.0, 0.3, 0.6, 1.0, 5.0, -3.0])
    def test_it_stays_inside_a_safe_range(self, intensity: float) -> None:
        # Bounded below 1.0 because Chatterbox's own guidance is that high
        # exaggeration degrades intelligibility — and a voice assistant
        # that cannot be understood is worse than one that is flat.
        # Out-of-range intensities are clamped, not trusted.
        value = exaggeration_for(state(EmotionLabel.HAPPY, intensity))
        assert 0.25 <= value <= 0.8

    def test_both_engines_derive_from_the_state_not_from_each_other(self) -> None:
        # Same NeiroState in, two independent outputs — so swapping TTS
        # at gate G5 changes one function, not the emotional contract.
        s = state(EmotionLabel.SAD, 0.8)
        assert isinstance(instruct_for(s), str)
        assert isinstance(exaggeration_for(s), float)
