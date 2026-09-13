"""Tests for the WER implementation (src/neiro/evals/wer.py).

This number decides every future STT swap for the life of the project,
so it's worth being sure it's actually right rather than plausible — a
WER implementation that's quietly too generous would make a worse model
look like an improvement.
"""

from __future__ import annotations

import math

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


class TestDevanagari:
    """Hindi is scored on the same terms as English, and this is where
    that claim is checked. Until 2026-09-14 the normaliser stripped
    every matra and nukta (see the comment above `_KEEP` in wer.py), so
    any Hindi WER it produced was a number about consonant fragments.
    """

    def test_matras_and_conjuncts_stay_attached_to_their_word(self) -> None:
        assert normalize("मेरे नाम यश है।") == ["मेरे", "नाम", "यश", "है"]

    def test_a_perfect_hindi_transcript_scores_zero(self) -> None:
        assert wer("मेरे नाम यश है।", "मेरे नाम यश है") == 0.0

    def test_danda_double_danda_and_abbreviation_sign_are_punctuation(self) -> None:
        assert normalize("राम॥ श्री॰ राम। ठीक है?") == ["राम", "श्री", "राम", "ठीक", "है"]

    def test_nukta_is_a_real_difference(self) -> None:
        # क़िला (qila) against किला (kila): the nukta changes the consonant,
        # and a normaliser that dropped it would call the two equal.
        assert wer("क़िला", "किला") == pytest.approx(1.0)

    def test_precomposed_and_decomposed_nukta_are_one_spelling(self) -> None:
        # U+0958 versus U+0915 U+093C — two encodings of the same word.
        assert wer("क़िला", "क़िला") == 0.0

    def test_candrabindu_and_anusvara_stay_on_the_word(self) -> None:
        assert normalize("हूँ हूं") == ["हूँ", "हूं"]

    def test_zero_width_joiners_do_not_split_a_word(self) -> None:
        assert normalize("रेल‌वे") == ["रेलवे"]
        assert wer("रेलवे", "रेल‌वे") == 0.0

    def test_devanagari_digits_are_not_expanded(self) -> None:
        # Same policy as "20" vs "twenty": a number that flatters the
        # model is worse than no number.
        assert normalize("२० मिनट") == ["२०", "मिनट"]
        assert wer("२० मिनट", "बीस मिनट") == pytest.approx(0.5)

    def test_code_switching_keeps_both_scripts(self) -> None:
        assert normalize("Neiro, गाना play करो!") == ["neiro", "गाना", "play", "करो"]

    def test_reference_word_count_is_words_not_fragments(self) -> None:
        # The denominator of a corpus WER. Fragmenting the reference
        # inflated it, which made every error look smaller than it was.
        result = score([("a", "मेरे नाम यश है", "मेरे नाम यश है")])
        assert result.total_ref_words == 4
        assert result.total_errors == 0


class TestNoTransliteration:
    def test_latin_script_hindi_against_devanagari_is_every_word_wrong(self) -> None:
        # The recogniser answered in the wrong script. That is the
        # failure a Hindi speaker actually hits, and it scores as one.
        assert wer("मैं ठीक हूँ", "main theek hoon") == pytest.approx(1.0)

    def test_hinglish_is_scored_as_the_latin_words_it_is(self) -> None:
        assert normalize("Yaar, kal ka plan kya hai?") == [
            "yaar",
            "kal",
            "ka",
            "plan",
            "kya",
            "hai",
        ]
        assert wer("yaar kal ka plan kya hai", "Yaar, kal ka plan kya hai?") == 0.0
        assert wer("yaar kal ka plan kya hai", "yaar kal ka plan kya tha") == pytest.approx(1 / 6)


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

    def test_empty_dataset_is_nan_not_zero(self) -> None:
        # It used to return 0.0, which does not divide by zero and is
        # worse: "WER 0.0%" is indistinguishable from a flawless run, and
        # a silently-empty dataset is exactly what produces it.
        result = DatasetScore(utterances=[])
        assert math.isnan(result.wer)
        assert result.total_ref_words == 0


class TestCorpusLevelNotMeanOfRates:
    """The distinction the benchmark exists to get right.

    A three-word utterance with one error scores 33%; a thirty-word
    utterance with one error scores 3%. Averaging those weights the short
    one ten times too heavily. Corpus WER is total errors over total
    reference words, which is what everyone means by the word.
    """

    def test_a_short_bad_utterance_does_not_dominate(self) -> None:
        pairs = [
            ("short", "yes it is", "no it is"),  # 1 error / 3 words  = 33%
            ("long", " ".join(["word"] * 30), " ".join(["word"] * 29 + ["wrong"])),  # 1/30 = 3%
        ]
        result = score(pairs)
        mean_of_rates = (1 / 3 + 1 / 30) / 2
        assert result.wer == pytest.approx(2 / 33, abs=1e-6)
        assert result.wer < mean_of_rates / 2

    def test_it_equals_total_errors_over_total_words(self) -> None:
        pairs = [("a", "one two three", "one two four"), ("b", "four five", "four five")]
        assert score(pairs).wer == pytest.approx(1 / 5)

    def test_an_empty_corpus_is_nan_not_a_perfect_score(self) -> None:
        # 0 errors over 0 words is not 0% WER, it is no measurement --
        # and 0.0 prints as "WER 0.0%", indistinguishable from a flawless
        # run, which is exactly what a silently-empty dataset produces.
        assert math.isnan(score([]).wer)

    def test_a_corpus_of_empty_references_is_also_nan(self) -> None:
        assert math.isnan(score([("a", "", "something")]).wer)


class TestBenchmarkHarness:
    def test_every_corpus_names_its_transcript_column(self) -> None:
        # Adding a corpus is a line in the table, not a change to the
        # harness — but a wrong column name would silently score against
        # empty references, which reads as 100% WER and looks like a
        # broken recogniser.
        import sys
        from pathlib import Path as P

        sys.path.insert(0, str(P(__file__).resolve().parents[1] / "scripts"))
        from bench_stt import CORPORA

        for name, spec in CORPORA.items():
            assert spec["text"] and spec["audio"], name
            assert spec["dir"], name

    def test_the_gate_is_about_his_voice_not_a_corpus(self) -> None:
        # A corpus number is a prior, not the gate. If this docstring
        # ever stops saying so, the benchmark will get mistaken for G3a.
        import sys
        from pathlib import Path as P

        sys.path.insert(0, str(P(__file__).resolve().parents[1] / "scripts"))
        import bench_stt

        assert "own voice" in bench_stt.__doc__
