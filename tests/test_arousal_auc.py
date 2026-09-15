"""Gate G3b's scorer — the number that decides the differentiator."""

from __future__ import annotations

import numpy as np

from elizabeth.evals.arousal_auc import GO_THRESHOLD, MIN_PER_SPEAKER, auc, score


class TestAuc:
    def test_perfect_separation(self) -> None:
        assert auc(np.array([1, 2, 3, 4.0]), np.array([False, False, True, True])) == 1.0

    def test_reversed_separation(self) -> None:
        assert auc(np.array([1, 2, 3, 4.0]), np.array([True, True, False, False])) == 0.0

    def test_a_constant_predictor_scores_chance_not_perfection(self) -> None:
        # Without averaging ties it would score 1.0, and a broken model
        # would look like a perfect one.
        assert auc(np.array([0.5] * 6), np.array([True, True, True, False, False, False])) == 0.5

    def test_one_class_only_is_undefined_not_zero(self) -> None:
        assert np.isnan(auc(np.array([1.0, 2.0]), np.array([True, True])))


class TestPerSpeaker:
    def test_it_scores_each_speaker_against_their_own_baseline(self) -> None:
        # A pooled AUC measures whether loud people differ from quiet
        # people. That is not the question.
        rows = []
        for speaker, offset in (("quiet", 0.0), ("loud", 100.0)):
            for i in range(MIN_PER_SPEAKER):
                high = i >= MIN_PER_SPEAKER // 2
                rows.append((speaker, offset + (2.0 if high else 1.0), high))
        s = score(rows)
        assert s.n_speakers == 2
        assert s.per_speaker_mean == 1.0

    def test_pooled_is_reported_so_the_gap_is_visible(self) -> None:
        # A quiet person's "excited" can be below a loud person's "calm",
        # which is exactly what pooling destroys.
        rows = [("quiet", 1.0, False), ("quiet", 2.0, True)] * 4
        rows += [("loud", 50.0, False), ("loud", 51.0, True)] * 4
        s = score(rows)
        assert s.pooled != s.per_speaker_mean

    def test_a_speaker_with_too_few_utterances_is_skipped(self) -> None:
        # An AUC over three clips is noise, not a measurement.
        rows = [("a", float(i), i > 1) for i in range(3)]
        assert score(rows).n_speakers == 0

    def test_a_speaker_with_one_class_is_skipped(self) -> None:
        rows = [("a", float(i), True) for i in range(10)]
        assert score(rows).n_speakers == 0


class TestVerdict:
    def test_the_threshold_is_the_plans(self) -> None:
        assert GO_THRESHOLD == 0.80

    def test_go_and_no_go(self) -> None:
        good = [("a", float(i), i >= 5) for i in range(10)]
        assert score(good).passed
        assert score(good).as_dict()["verdict"] == "GO"

        rng = np.random.default_rng(0)
        noise = [("a", float(rng.random()), i >= 5) for i in range(40)]
        assert score(noise).as_dict()["verdict"] in {"GO", "NO-GO"}

    def test_valence_is_reported_and_not_gated(self) -> None:
        # Measured at 0.511 on 30 speakers — chance. Printed so the
        # asymmetry stays visible rather than quietly dropped when it
        # disappoints.
        rows = [("a", float(i), i >= 5) for i in range(10)]
        val = [("a", 0.5, i >= 5) for i in range(10)]
        out = score(rows, valence_rows=val).as_dict()
        assert out["valence_auc_per_speaker_mean"] == 0.5
        assert out["verdict"] == "GO"  # valence did not affect it

    def test_worst_speaker_is_reported(self) -> None:
        # A mean of 0.84 hiding a speaker at 0.55 matters when the
        # speaker who matters is one person.
        rows = [("good", float(i), i >= 5) for i in range(10)]
        rows += [("bad", 1.0, i >= 5) for i in range(10)]
        s = score(rows)
        assert s.worst_speaker < s.best_speaker

    def test_no_usable_speakers_gives_nan_not_a_pass(self) -> None:
        s = score([("a", 1.0, True)])
        assert np.isnan(s.per_speaker_mean)
        assert not s.passed
