"""IPA phonemes to VRM mouth shapes.

VRM 1.0 spells its mouth presets `aa ih ou ee oh`. VRM 0.x spelled them
`A I U E O`. Using the wrong set makes `expressionManager.setValue()`
silently do nothing — the single most common reason an avatar's mouth
stays shut with no error anywhere.
"""

from __future__ import annotations

import pytest

from neiro.tts.visemes import SILENCE, VISEMES, timeline, viseme_for


class TestPresetNames:
    def test_the_names_are_vrm_1_0_not_0_x(self) -> None:
        assert VISEMES == ("aa", "ih", "ou", "ee", "oh")
        assert not any(v.isupper() for v in VISEMES)

    def test_every_mapping_lands_on_a_real_preset(self) -> None:
        for phoneme in "ɑaæʌiɪuʊeɛoɔmbpsztdnlrwfvkɡŋ":
            shape = viseme_for(phoneme)
            assert shape in VISEMES, f"{phoneme} -> {shape}"


class TestMapping:
    @pytest.mark.parametrize(
        ("phoneme", "expected"),
        [("ɑ", "aa"), ("i", "ih"), ("u", "ou"), ("ɛ", "ee"), ("oʊ", "oh")],
    )
    def test_the_vowels_land_where_a_mouth_would(self, phoneme: str, expected: str) -> None:
        assert viseme_for(phoneme) == expected

    def test_rounded_consonants_use_a_rounded_shape(self) -> None:
        assert viseme_for("w") == "ou"
        assert viseme_for("m") == "oh"

    @pytest.mark.parametrize("symbol", [" ", ".", ",", "?", "!", "", "…"])
    def test_silence_and_punctuation_close_the_mouth(self, symbol: str) -> None:
        assert viseme_for(symbol) == SILENCE

    def test_an_unknown_phoneme_keeps_the_mouth_moving(self) -> None:
        # A mouth that freezes mid-word reads as a bug; a slightly wrong
        # shape does not. So unknowns fall back to a shape, not silence.
        assert viseme_for("ʡ") in VISEMES
        assert viseme_for("ʘ") in VISEMES

    def test_a_length_mark_falls_back_to_its_base_vowel(self) -> None:
        assert viseme_for("ɑː") == viseme_for("ɑ")


class TestTimeline:
    def test_events_are_in_order_and_start_at_the_offset(self) -> None:
        events = timeline("hɛloʊ", [0.05, 0.08, 0.06, 0.12, 0.09], offset=1.5)
        assert events[0][0] == pytest.approx(1.5)
        starts = [t for t, _, _ in events]
        assert starts == sorted(starts)

    def test_repeated_shapes_are_merged(self) -> None:
        # "ss" is one long `ih`, not two. Re-triggering the same
        # blendshape produces a visible stutter on the face.
        events = timeline("ss", [0.1, 0.1])
        assert len(events) == 1
        assert events[0][2] == pytest.approx(0.2)

    def test_different_shapes_are_not_merged(self) -> None:
        # "s" and "i" BOTH map to `ih` (narrow, unrounded), so they
        # correctly merge — the first version of this test used them and
        # was asserting the opposite of the mapping. Use two phonemes
        # that genuinely differ.
        assert viseme_for("s") == viseme_for("i") == "ih"
        assert len(timeline("sɑ", [0.1, 0.1])) == 2
        assert len(timeline("mi", [0.1, 0.1])) == 2

    def test_total_duration_is_preserved_by_merging(self) -> None:
        durations = [0.05, 0.05, 0.1, 0.1, 0.2]
        events = timeline("ssiiɑ", durations)
        assert sum(d for _, _, d in events) == pytest.approx(sum(durations))

    def test_mismatched_lengths_do_not_raise(self) -> None:
        # Real engines occasionally disagree by one; dropping the extra
        # beats crashing the turn.
        assert timeline("abc", [0.1])
        assert timeline("a", [0.1, 0.2, 0.3])

    def test_empty_input(self) -> None:
        assert timeline("", []) == []
