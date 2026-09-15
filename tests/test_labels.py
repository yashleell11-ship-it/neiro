"""The emotion label contract shared by training and runtime.

The one that matters: an unrecognised label maps to nothing and gets
DROPPED, never quietly to neutral. Training every ambiguous utterance as
flat would teach the model that uncertainty sounds calm — which is the
opposite of true, and would make her worst at exactly the moments that
matter.
"""

from __future__ import annotations

import pytest

from elizabeth.affect.labels import (
    ACTED_CORPORA,
    ALIASES,
    CIRCUMPLEX,
    KINDS,
    NATURAL_CORPORA,
    SYNTHETIC_CORPORA,
    is_acted,
    normalise,
    speech_kind,
    to_circumplex,
)


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ANG", "anger"),  # CREMA-D filename code
            ("ang", "anger"),
            ("Angry", "anger"),
            ("angry", "anger"),
            ("happiness", "happy"),
            ("joy", "happy"),
            ("sadness", "sad"),
            ("fearful", "fear"),
            ("ps", "surprise"),  # TESS pleasant-surprise code
            ("pleasant surprise", "surprise"),
            ("sleepy", "tired"),
            ("fru", "frustrated"),
            ("  Neutral  ", "neutral"),
        ],
    )
    def test_every_corpus_spelling_lands_on_one_name(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    @pytest.mark.parametrize("raw", ["xxx", "other", "unknown", "none", "banana", "", "   "])
    def test_unusable_labels_are_dropped_not_called_neutral(self, raw: str) -> None:
        # The whole point of this file. Dropping loses an example;
        # mislabelling it teaches the model something false.
        assert normalise(raw) == ""
        assert to_circumplex(raw) is None

    def test_neutral_is_not_the_same_as_unusable(self) -> None:
        assert normalise("neutral") == "neutral"
        assert to_circumplex("neutral") is not None


class TestCircumplex:
    def test_all_coordinates_are_in_range(self) -> None:
        for name, point in CIRCUMPLEX.items():
            assert -1.0 <= point.valence <= 1.0, name
            assert -1.0 <= point.arousal <= 1.0, name

    def test_neutral_is_the_origin(self) -> None:
        # Everything else is defined relative to it, so it cannot drift.
        assert CIRCUMPLEX["neutral"].valence == 0.0
        assert CIRCUMPLEX["neutral"].arousal == 0.0

    def test_the_axes_point_the_established_way(self) -> None:
        # Arousal: anger and fear are activated; tired and bored are not.
        assert CIRCUMPLEX["anger"].arousal > 0.5
        assert CIRCUMPLEX["fear"].arousal > 0.5
        assert CIRCUMPLEX["tired"].arousal < -0.5
        assert CIRCUMPLEX["bored"].arousal < -0.5
        # Valence: happy is positive, sad and anger negative.
        assert CIRCUMPLEX["happy"].valence > 0.5
        assert CIRCUMPLEX["sad"].valence < 0
        assert CIRCUMPLEX["anger"].valence < 0

    def test_anger_and_sad_differ_mainly_on_arousal(self) -> None:
        # This separation is the one Lane A can actually hear, and the
        # one gate G3b measures. If the label geometry ever collapses it,
        # the gate is measuring nothing.
        anger, sad = CIRCUMPLEX["anger"], CIRCUMPLEX["sad"]
        assert abs(anger.arousal - sad.arousal) > abs(anger.valence - sad.valence)

    def test_every_alias_target_exists_or_is_a_deliberate_drop(self) -> None:
        for raw, target in ALIASES.items():
            assert target == "" or target in CIRCUMPLEX, f"{raw} -> {target}"


class TestActedVsNatural:
    def test_the_distinction_is_recorded_for_the_corpora_in_use(self) -> None:
        assert is_acted("crema-d") is True
        assert is_acted("ravdess") is True
        assert is_acted("msp-podcast") is False
        assert is_acted("iemocap") is False

    def test_prefix_matching_handles_versioned_names(self) -> None:
        # The manifest slugs are things like "msp-podcast-v2-0".
        assert is_acted("msp-podcast-v2-0") is False
        assert is_acted("crema-d") is True

    def test_an_unrecorded_corpus_returns_none_not_a_guess(self) -> None:
        # Reporting one accuracy over a mix of acted and natural speech
        # without saying which is the commonest way SER results mislead.
        # None keeps that visible instead of defaulting to a claim.
        assert is_acted("some-new-corpus") is None

    def test_the_two_sets_do_not_overlap(self) -> None:
        assert not (ACTED_CORPORA & NATURAL_CORPORA)


class TestEmoNetSpellings:
    """EmoNet-Voice's 42 categories, as the corpus spells them after
    URL-decoding, each landing on one circumplex name or on nothing."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Impatience and Irritability", "frustrated"),
            ("Jealousy & Envy", "frustrated"),
            ("Astonishment", "surprise"),
            ("Fatigue", "tired"),
            ("Thankfulness", "grateful"),
            ("Contentment", "content"),
            ("Distress", "distressed"),
            ("Pain", "distressed"),
            ("Teasing", "amused"),
            ("Sadness", "sad"),
            ("Helplessness", "sad"),
            ("Triumph", "excited"),
            ("Shame", "ashamed"),
            ("Awe", "awe"),
        ],
    )
    def test_each_category_lands_on_one_name(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Authenticity",
            "Arousal",
            "Concentration",
            "Contemplation",
            "Intoxication",
            "Emotional Numbness",
            "Sexual Lust",
        ],
    )
    def test_non_emotions_are_dropped_not_placed(self, raw: str) -> None:
        # "Arousal" is the axis, not a point on it; "Authenticity" is a
        # judgement about the clip. Placing either would train a
        # coordinate nobody chose.
        assert normalise(raw) == ""
        assert to_circumplex(raw) is None

    def test_the_new_points_sit_in_their_quadrants(self) -> None:
        distressed = to_circumplex("Distress")
        assert distressed is not None and distressed.valence < 0 and distressed.arousal > 0.5
        content = to_circumplex("Contentment")
        assert content is not None and content.valence > 0 and content.arousal < 0
        embarrassed = to_circumplex("Embarrassment")
        assert embarrassed is not None and embarrassed.valence < 0 and embarrassed.arousal > 0
        disappointed = to_circumplex("Disappointment")
        assert disappointed is not None and disappointed.valence < 0 and disappointed.arousal < 0
        hopeful = to_circumplex("Hope")
        assert hopeful is not None and hopeful.valence > 0 and hopeful.arousal > 0

    def test_distress_and_contentment_differ_on_both_axes(self) -> None:
        # The pair the corpus adds that the acted sets never had: one is
        # negative and activated, the other positive and deactivated. If
        # they ever collapse toward each other the corpus stops adding
        # anything the model could learn.
        a, b = to_circumplex("Distress"), to_circumplex("Contentment")
        assert a is not None and b is not None
        assert abs(a.valence - b.valence) > 1.0
        assert abs(a.arousal - b.arousal) > 0.5


class TestSpeechKind:
    def test_synthetic_is_neither_acted_nor_natural(self) -> None:
        # A TTS engine's idea of anger must not be averaged into the
        # acted number, and certainly not into the natural one.
        assert speech_kind("emonet-voice-bench") == "synthetic"
        assert is_acted("emonet-voice-bench") is None

    def test_kind_agrees_with_the_two_way_answer(self) -> None:
        assert speech_kind("crema-d") == "acted" and is_acted("crema-d") is True
        assert (
            speech_kind("msp-podcast-v2-0") == "natural" and is_acted("msp-podcast-v2-0") is False
        )
        assert speech_kind("some-new-corpus") is None and is_acted("some-new-corpus") is None

    def test_every_kind_is_reachable(self) -> None:
        # The recipe iterates KINDS to build its per-kind report; a kind
        # no corpus can ever produce would be an empty column forever.
        for kind, members in (
            ("acted", ACTED_CORPORA),
            ("natural", NATURAL_CORPORA),
            ("synthetic", SYNTHETIC_CORPORA),
        ):
            assert kind in KINDS
            assert all(speech_kind(name) == kind for name in members)

    def test_the_three_sets_do_not_overlap(self) -> None:
        assert not (ACTED_CORPORA & SYNTHETIC_CORPORA)
        assert not (NATURAL_CORPORA & SYNTHETIC_CORPORA)
