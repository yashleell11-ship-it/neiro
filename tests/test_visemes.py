"""IPA phonemes to VRM mouth shapes.

VRM 1.0 spells its mouth presets `aa ih ou ee oh`. VRM 0.x spelled them
`A I U E O`. Using the wrong set makes `expressionManager.setValue()`
silently do nothing — the single most common reason an avatar's mouth
stays shut with no error anywhere.

The strings below are what misaki (Kokoro's G2P) really emits — checked
by running `misaki.en.G2P` on the words in the comments — not textbook
IPA. Kokoro's `pred_dur` is one duration per character of that string,
so the mapping has to be right one character at a time.
"""

from __future__ import annotations

import pytest

from neiro.tts.visemes import SILENCE, VISEMES, timeline, viseme_for

# misaki/en.py `US_VOCAB`: every symbol its American-English G2P can
# emit inside a word. Copied, not imported — importing misaki drags in
# spaCy, and the point is that the map covers this alphabet even when
# the G2P is not installed.
MISAKI_US_VOCAB = "AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ"
# misaki/en.py `GB_VOCAB` adds the length mark and `Q` (əʊ); the US
# path strips `ː` in espeak.py but the map must not depend on that.
MISAKI_GB_EXTRA = "Qː"
STRESS_MARKS = "ˈˌ"


class TestPresetNames:
    def test_the_names_are_vrm_1_0_not_0_x(self) -> None:
        assert VISEMES == ("aa", "ih", "ou", "ee", "oh")
        assert not any(v.isupper() for v in VISEMES)

    def test_every_mapping_lands_on_a_real_preset(self) -> None:
        for phoneme in "ɑaæʌiɪuʊeɛoɔmbpsztdnlrwfvkɡŋ":
            shape = viseme_for(phoneme)
            assert shape in VISEMES, f"{phoneme} -> {shape}"

    def test_every_symbol_misaki_emits_lands_on_a_real_preset(self) -> None:
        for symbol in MISAKI_US_VOCAB + MISAKI_GB_EXTRA:
            if symbol in STRESS_MARKS or symbol == "ː":
                continue
            shape = viseme_for(symbol)
            assert shape in VISEMES, f"{symbol!r} -> {shape!r}"


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

    @pytest.mark.parametrize("symbol", [" ", ".", ",", "?", "!", "", "…", "“", "”", "(", ")"])
    def test_silence_and_punctuation_close_the_mouth(self, symbol: str) -> None:
        assert viseme_for(symbol) == SILENCE

    def test_an_unknown_phoneme_keeps_the_mouth_moving(self) -> None:
        # A mouth that freezes mid-word reads as a bug; a slightly wrong
        # shape does not. So unknowns fall back to a shape, not silence.
        assert viseme_for("ʡ") in VISEMES
        assert viseme_for("ʘ") in VISEMES


class TestMisakiAlphabet:
    """misaki writes diphthongs and affricates as one letter each, so
    that one character of the phoneme string is one `pred_dur` slot.
    Each letter must render as the two-character IPA it stands for."""

    @pytest.mark.parametrize(
        ("letter", "ipa"),
        [
            ("A", "eɪ"),  # "day"   -> dˈA
            ("I", "aɪ"),  # "I'm"   -> ˌIm
            ("O", "oʊ"),  # "hello" -> həlˈO
            ("W", "aʊ"),  # "how"   -> hˌW
            ("Y", "ɔɪ"),  # "boy"   -> bˈY
            ("ʧ", "tʃ"),  # "church" -> ʧˈɜɹʧ
            ("ʤ", "dʒ"),  # "judge" -> ʤˈʌʤ
        ],
    )
    def test_a_single_letter_renders_as_the_ipa_it_stands_for(self, letter: str, ipa: str) -> None:
        assert viseme_for(letter) == viseme_for(ipa)

    @pytest.mark.parametrize(
        ("letter", "expected"),
        [
            ("O", "oh"),  # was "aa": "hello" ended with the jaw dropping open
            ("Y", "oh"),  # was "aa": "boy" opened wide instead of rounding
            ("Q", "oh"),  # GB "go" -> ɡˌQ
            ("A", "ee"),
            ("I", "aa"),
            ("W", "aa"),
            ("ʧ", "ih"),
            ("ʤ", "ih"),
            ("T", "ih"),  # "butter" -> bˈʌTəɹ, the flap
            ("ɾ", "ih"),
            ("ᵻ", "ih"),  # "roses" -> ɹˈOzᵻz
            ("ᵊ", "aa"),  # "bottle" -> bˈɑtᵊl, syllabic schwa
            ("ʔ", "aa"),
        ],
    )
    def test_the_misaki_letters_have_the_right_mouth(self, letter: str, expected: str) -> None:
        assert viseme_for(letter) == expected

    def test_uppercase_is_not_a_vrm_0_x_preset_name(self) -> None:
        # `A I O` in misaki's alphabet are phonemes. They must never be
        # confused with VRM 0.x's `A I U E O` mouth names and passed
        # through as shapes.
        for letter in "AIOWYQ":
            assert viseme_for(letter) in VISEMES


class TestModifiers:
    """Length and stress marks describe the phone next to them and have
    no shape of their own. Kokoro gives each one a duration anyway, so
    they must extend the previous shape rather than emit one."""

    @pytest.mark.parametrize("mark", ["ː", "ˑ", "ˈ", "ˌ", "̃", "͡", "̩", "˞"])
    def test_a_bare_modifier_holds_the_previous_shape(self, mark: str) -> None:
        assert viseme_for(mark, "ou") == "ou"
        assert viseme_for(mark, "oh") == "oh"
        assert viseme_for(mark, "ih") == "ih"

    @pytest.mark.parametrize("mark", ["ː", "ˈ", "ˌ"])
    def test_a_modifier_with_nothing_before_it_is_silence(self, mark: str) -> None:
        assert viseme_for(mark) == SILENCE

    @pytest.mark.parametrize(
        ("phoneme", "expected"),
        [
            ("uː", "ou"),  # GB "food"
            ("ɔː", "oh"),  # GB "thought"
            ("ɑː", "aa"),
            ("ˈu", "ou"),
            ("ɔ̃", "oh"),  # nasalised, base ɔ
            ("t͡ʃ", "ih"),  # tie-barred affricate
        ],
    )
    def test_a_marked_phoneme_renders_as_its_base(self, phoneme: str, expected: str) -> None:
        assert viseme_for(phoneme) == expected

    def test_the_syllabic_schwa_is_a_sound_not_a_modifier(self) -> None:
        # Unicode files `ᵊ` under "modifier letter"; misaki uses it as
        # the vowel in "bottle". The map must win over the category.
        assert viseme_for("ᵊ", "ou") == "aa"


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


class TestTimelineOnMisakiStrings:
    """The strings Kokoro actually hands `timeline()`, with one duration
    per character. The bug these guard: a stress or length mark looked up
    as a phoneme fell through to `aa`, and the jaw snapped open in the
    middle of a rounded vowel."""

    def test_food_is_two_shapes_with_no_open_jaw_between(self) -> None:
        # "food" -> fˈud. f and u are both rounded (`ou`); the stress
        # mark sits between them and must not split them.
        events = timeline("fˈud", [0.05, 0.0125, 0.15, 0.05])
        assert [shape for _, shape, _ in events] == ["ou", "ih"]
        assert events[0][2] == pytest.approx(0.05 + 0.0125 + 0.15)

    def test_a_length_mark_extends_the_held_vowel(self) -> None:
        # GB "food" -> fˈuːd. The `ː` slot is the held part of the vowel
        # and the lips stay rounded through it.
        events = timeline("fˈuːd", [0.05, 0.0125, 0.1, 0.08, 0.05])
        assert [shape for _, shape, _ in events] == ["ou", "ih"]
        assert events[0][2] == pytest.approx(0.05 + 0.0125 + 0.1 + 0.08)

    def test_thought_holds_the_rounded_vowel_through_its_length_mark(self) -> None:
        # GB "thought" -> θˈɔːt: narrow, rounded-open, narrow. The `ː`
        # belongs to the `oh` event, not to a fresh `aa` between ɔ and t.
        events = timeline("θˈɔːt", [0.05, 0.0125, 0.1, 0.08, 0.05])
        assert [shape for _, shape, _ in events] == ["ih", "oh", "ih"]
        assert events[1][2] == pytest.approx(0.1 + 0.08)

    def test_the_reported_case_um_with_a_long_vowel(self) -> None:
        # The review's reproduction: `uːm` rendered ou, aa, oh.
        events = timeline("uːm", [0.1, 0.1, 0.1])
        assert [shape for _, shape, _ in events] == ["ou", "oh"]
        assert events[0][2] == pytest.approx(0.2)

    def test_hello_ends_rounded_not_open(self) -> None:
        # "hello" -> həlˈO. `O` is oʊ and the word ends on a rounded
        # mouth; it used to end on the open-jaw fallback.
        events = timeline("həlˈO", [0.04, 0.05, 0.06, 0.0125, 0.15])
        assert [shape for _, shape, _ in events] == ["aa", "ih", "oh"]
        assert events[-1][1] == "oh"

    def test_a_leading_stress_mark_is_silence_until_the_word_starts(self) -> None:
        # "I'm" -> ˌIm. Nothing precedes the mark, so the mouth is
        # closed for its slot — it had not opened yet.
        events = timeline("ˌIm", [0.0125, 0.12, 0.08])
        assert [shape for _, shape, _ in events] == [SILENCE, "aa", "oh"]
        assert events[0][2] == pytest.approx(0.0125)
        assert events[1][0] == pytest.approx(0.0125)

    def test_a_full_misaki_sentence_never_hits_the_fallback_by_accident(self) -> None:
        # "I'm so happy!" -> ˌIm sˌO hˈæpi!  Every `aa` in the result
        # must come from a real open vowel (I, æ), never from a modifier.
        phonemes = "ˌIm sˌO hˈæpi!"
        events = timeline(phonemes, [0.05] * len(phonemes))
        assert [shape for _, shape, _ in events] == [
            SILENCE,  # ˌ
            "aa",  # I
            "oh",  # m
            SILENCE,  # space
            "ih",  # s + ˌ
            "oh",  # O
            SILENCE,  # space
            "aa",  # h + ˈ + æ
            "oh",  # p
            "ih",  # i
            SILENCE,  # !
        ]
        assert sum(d for _, _, d in events) == pytest.approx(0.05 * len(phonemes))

    def test_total_duration_survives_modifiers(self) -> None:
        durations = [0.03, 0.01, 0.12, 0.09, 0.04]
        events = timeline("θˈɔːt", durations)
        assert sum(d for _, _, d in events) == pytest.approx(sum(durations))
