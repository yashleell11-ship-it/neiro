"""Tests for the WER implementation (src/neiro/evals/wer.py).

This number decides every future STT swap for the life of the project,
so it's worth being sure it's actually right rather than plausible — a
WER implementation that's quietly too generous would make a worse model
look like an improvement.
"""

from __future__ import annotations

import pytest

from neiro.evals.wer import DatasetScore, edit_distance, normalize, score, wer


class TestNormalize:
    def test_lowercases_and_splits(self) -> None:
        assert normalize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert normalize("Hello, world! How are you?") == ["hello", "world", "how", "are", "you"]

    def test_collapses_whitespace(self) -> None:
        assert normalize("  hello   \n world  ") == ["hello", "world"]

    def test_keeps_apostrophes(self) -> None:
        # "don't" vs "dont" is a real transcription difference; collapsing
        # it would flatter the model.
        assert normalize("don't") == ["don't"]

    def test_empty_string_is_no_words(self) -> None:
        assert normalize("") == []
        assert normalize("   ") == []
        assert normalize("!!!") == []

    def test_does_not_expand_numerals(self) -> None:
        # Deliberate: "20" vs "twenty" stays an error. A number that
        # flatters the model is worse than no number.
        assert normalize("set it to 20") == ["set", "it", "to", "20"]


class TestEditDistance:
    def test_identical_is_zero(self) -> None:
        assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_one_substitution(self) -> None:
        assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_one_deletion(self) -> None:
        assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1

    def test_one_insertion(self) -> None:
        assert edit_distance(["a", "c"], ["a", "b", "c"]) == 1

    def test_empty_reference(self) -> None:
        assert edit_distance([], ["a", "b"]) == 2

    def test_empty_hypothesis(self) -> None:
        assert edit_distance(["a", "b"], []) == 2

    def test_both_empty(self) -> None:
        assert edit_distance([], []) == 0

    def test_completely_different(self) -> None:
        assert edit_distance(["a", "b"], ["x", "y"]) == 2


class TestWer:
    def test_perfect_transcription_is_zero(self) -> None:
        assert wer("hello world", "hello world") == 0.0

    def test_punctuation_and_case_are_ignored(self) -> None:
        assert wer("Hello, world!", "hello world") == 0.0

    def test_one_wrong_word_in_four(self) -> None:
        assert wer("the quick brown fox", "the quick brown dog") == pytest.approx(0.25)

    def test_dropped_word(self) -> None:
        assert wer("the quick brown fox", "the quick fox") == pytest.approx(0.25)

    def test_empty_hypothesis_is_total_failure(self) -> None:
        # Every reference word is a deletion — this is what a rejected
        # turn (the STT confidence floor returning "") scores as.
        assert wer("hello world", "") == pytest.approx(1.0)

    def test_wer_can_exceed_one(self) -> None:
        # A hypothesis longer and wrong scores above 1.0. That is correct
        # and is precisely why the corpus aggregate is computed from raw
        # error counts rather than by averaging these.
        assert wer("hello", "completely different words here now") > 1.0

    def test_empty_reference_with_output_is_an_error(self) -> None:
        assert wer("", "hallucinated text") == pytest.approx(1.0)

    def test_empty_reference_with_no_output_is_perfect(self) -> None:
        assert wer("", "") == 0.0


class TestDatasetScore:
    def test_aggregate_is_corpus_level_not_mean_of_utterances(self) -> None:
        # THE distinction that matters. One short utterance fully wrong,
        # one long utterance fully right.
        #   corpus-level: 1 error / 11 ref words     = 0.0909...
        #   mean of per-utterance WERs: (1.0 + 0.0)/2 = 0.5
        # Averaging would weight a one-word command the same as a
        # ten-word sentence and report ~5x the real error rate.
        result = score(
            [
                ("short", "yes", "no"),
                (
                    "long",
                    "the quick brown fox jumps over the lazy dog today",
                    "the quick brown fox jumps over the lazy dog today",
                ),
            ]
        )
        assert result.total_ref_words == 11
        assert result.total_errors == 1
        assert result.wer == pytest.approx(1 / 11)

        mean_of_utterances = sum(u.wer for u in result.utterances) / len(result.utterances)
        assert mean_of_utterances == pytest.approx(0.5)
        assert result.wer != pytest.approx(mean_of_utterances)

    def test_per_utterance_scores_are_available(self) -> None:
        result = score([("a", "hello world", "hello there")])
        assert len(result.utterances) == 1
        assert result.utterances[0].name == "a"
        assert result.utterances[0].errors == 1
        assert result.utterances[0].ref_words == 2
        assert result.utterances[0].wer == pytest.approx(0.5)

    def test_perfect_dataset_is_zero(self) -> None:
        result = score([("a", "hello", "hello"), ("b", "world", "world")])
        assert result.wer == 0.0

    def test_empty_dataset_does_not_divide_by_zero(self) -> None:
        result = DatasetScore(utterances=[])
        assert result.wer == 0.0
        assert result.total_ref_words == 0
