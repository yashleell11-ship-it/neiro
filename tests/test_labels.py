"""The emotion label contract shared by training and runtime.

The one that matters: an unrecognised label maps to nothing and gets
DROPPED, never quietly to neutral. Training every ambiguous utterance as
flat would teach the model that uncertainty sounds calm — which is the
opposite of true, and would make her worst at exactly the moments that
matter.
"""

from __future__ import annotations

import pytest

from neiro.affect.labels import (
    ACTED_CORPORA,
    ALIASES,
    CIRCUMPLEX,
    NATURAL_CORPORA,
    is_acted,
    normalise,
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
