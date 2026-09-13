"""Scoring the endpointer. Two bars that fail in opposite directions."""

from __future__ import annotations

import math

from neiro.evals.endpoint import MAX_FALSE_CUT_RATE, TARGET_P50_MS, percentile, score


def ends(delays: list[float]) -> list[tuple[bool, bool, float]]:
    return [(True, True, d) for d in delays]


class TestTwoBars:
    def test_fast_but_cutting_people_off_fails(self) -> None:
        # Speed is not the only axis. A system that ends every turn
        # instantly and cuts a fifth of them is not a good endpointer.
        decisions = ends([50.0] * 10) + [(True, False, 0.0)] * 5
        s = score(decisions)
        assert s.p50_ms < TARGET_P50_MS
        assert not s.passed

    def test_accurate_but_slow_fails_too(self) -> None:
        s = score(ends([2000.0] * 10))
        assert s.false_cut_rate == 0.0
        assert not s.passed

    def test_both_bars_met_passes(self) -> None:
        decisions = ends([150.0] * 20) + [(False, False, 0.0)] * 20
        assert score(decisions).passed

    def test_the_two_are_never_averaged_into_one_accuracy(self) -> None:
        out = score(ends([150.0] * 5)).as_dict()
        assert "accuracy" not in out
        assert {"p50_ms", "false_cut_rate"} <= set(out)


class TestPercentiles:
    def test_acceptance_is_on_p50_with_p90_alongside(self) -> None:
        # A mean would let a 3-second hang vanish behind fast turns, and
        # the hang is what feels broken.
        #
        # Two slow turns in ten, not one: under nearest-rank the p90 of
        # ten samples IS the 9th, so a single outlier at the top is p100
        # and p90 correctly still reads 100. An earlier version of this
        # test used one outlier and only passed because the percentile
        # was off by one.
        s = score(ends([100.0] * 8 + [3000.0, 3000.0]))
        assert s.p50_ms == 100.0
        assert s.p90_ms == 3000.0
        assert "mean" not in str(s.as_dict())

    def test_one_outlier_in_ten_does_not_move_p90(self) -> None:
        # The nearest-rank consequence, stated so it is not mistaken for
        # a bug later.
        s = score(ends([100.0] * 9 + [3000.0]))
        assert s.p90_ms == 100.0

    def test_the_definition_matches_the_latency_eval(self) -> None:
        # One definition of a percentile across the project, or two
        # numbers stop being comparable.
        from neiro.evals import latency

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0.5) == latency.percentile(values, 0.5)
        assert percentile(values, 0.95) == latency.percentile(values, 0.95)


class TestHesitations:
    def test_they_are_scored_separately(self) -> None:
        # "I want to go to... uh... the library" is the whole point of
        # having a semantic endpointer. A system perfect on ordinary
        # speech that cuts every hesitation is worse than one slightly
        # slower at both.
        s = score(ends([150.0] * 10), hesitations=[True, True, False, True])
        assert s.hesitations_survived == 0.75

    def test_absent_hesitations_report_none_not_a_perfect_score(self) -> None:
        assert score(ends([150.0] * 5)).hesitations_survived is None


class TestEdges:
    def test_missed_endpoints_are_counted(self) -> None:
        # Never firing is its own failure: max_wait catches it, but the
        # rate says how often the model abdicated.
        decisions = [(False, True, 0.0)] * 4 + [(True, True, 150.0)] * 6
        assert score(decisions).missed_rate == 0.4

    def test_no_data_is_nan_not_a_pass(self) -> None:
        s = score([])
        assert math.isnan(s.p50_ms)
        assert not s.passed

    def test_the_thresholds_are_the_plans(self) -> None:
        assert TARGET_P50_MS == 235.0
        assert MAX_FALSE_CUT_RATE <= 0.05
