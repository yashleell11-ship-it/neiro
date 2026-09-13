"""The one metric.

Three rules, each easy to break by accident and none of which produce an
error when broken — so each gets a test.
"""

from __future__ import annotations

import json

import pytest

from neiro.evals.latency import (
    HEADLINE_END,
    headline_ms,
    percentile,
    read_turns,
    stage_breakdown,
    summarise,
)


class TestNoMeans:
    def test_the_summary_has_no_mean_at_all(self) -> None:
        # Deliberately absent, not merely unused. A mean hides the
        # four-second turn behind nineteen good ones, and the four-second
        # turn is the one that makes her feel broken.
        out = summarise([{"latency_ms": v} for v in (800, 900, 1000, 4000)]).as_dict()
        assert "mean" not in json.dumps(out)
        assert {"p50_ms", "p95_ms", "worst_ms"} <= set(out)

    def test_p95_surfaces_the_bad_turn_that_p50_hides(self) -> None:
        s = summarise([{"latency_ms": v} for v in (820, 900, 950, 1010, 1100, 1180, 4200)])
        assert s.p50_ms < 1100
        assert s.p95_ms > 4000


class TestPercentile:
    def test_it_does_not_interpolate(self) -> None:
        # Interpolating invents a latency nobody experienced. With twenty
        # turns the honest p95 is the second-worst that actually happened.
        values = [100.0, 200.0, 300.0, 400.0]
        assert percentile(values, 0.5) in values
        assert percentile(values, 0.95) in values

    def test_edges(self) -> None:
        assert percentile([], 0.5) != percentile([], 0.5)  # nan
        assert percentile([42.0], 0.95) == 42.0

    def test_p100_is_the_worst(self) -> None:
        assert percentile([1.0, 9.0, 5.0], 1.0) == 9.0


class TestHeadline:
    def test_it_ends_at_the_browsers_callback(self) -> None:
        # Everything before that is "we sent it".
        assert HEADLINE_END == "sink_played"

    def test_it_measures_endpoint_to_played(self) -> None:
        assert headline_ms({"endpoint": 1.0, "sink_played": 2.0}) == pytest.approx(1000.0)

    def test_a_turn_with_no_audio_is_none_not_zero(self) -> None:
        # A cancelled or failed turn has no latency. Recording 0.0 would
        # drag every percentile toward a number that never happened.
        assert headline_ms({"endpoint": 1.0}) is None
        assert headline_ms({}) is None

    def test_a_file_sink_stamp_does_not_count_as_played(self) -> None:
        # sink_written_seq0 is named differently on purpose: a file has
        # no moment when sound left a speaker, and averaging the two
        # would quietly redefine the metric.
        assert headline_ms({"endpoint": 1.0, "sink_written_seq0": 2.0}) is None

    def test_no_audio_turns_are_excluded_from_percentiles(self) -> None:
        s = summarise(
            [{"timeline": {"endpoint": 1.0, "sink_played": 2.0}}, {"timeline": {"endpoint": 5.0}}]
        )
        assert s.n == 1


class TestTiersStaySeparate:
    def test_a_summary_covers_one_tier_and_one_profile(self) -> None:
        # A p50 over a mix of laptop and box turns describes neither.
        records = [
            {"latency_ms": 900, "tier": "local", "profile": "earbuds"},
            {"latency_ms": 200, "tier": "lan", "profile": "earbuds"},
            {"latency_ms": 950, "tier": "local", "profile": "speakers"},
        ]
        assert summarise(records, tier="local", profile="earbuds").n == 1
        assert summarise(records, tier="lan", profile="earbuds").p50_ms == 200

    def test_the_summary_names_what_it_describes(self) -> None:
        out = summarise([{"latency_ms": 900}], tier="lan", profile="speakers").as_dict()
        assert out["tier"] == "lan" and out["profile"] == "speakers"


class TestWaterfall:
    def test_it_names_consecutive_stages(self) -> None:
        # What turns "it feels slow" into "STT is 140 ms, the LLM is 15 s".
        w = stage_breakdown(
            {"endpoint": 0.0, "stt_start": 0.0, "stt_done": 0.141, "sink_played": 1.12}
        )
        assert w["stt_start->stt_done"] == pytest.approx(141, abs=1)

    def test_missing_stages_are_skipped_not_zeroed(self) -> None:
        w = stage_breakdown({"endpoint": 0.0, "sink_played": 1.0})
        assert list(w) == ["endpoint->sink_played"]

    def test_an_empty_timeline_gives_nothing(self) -> None:
        assert stage_breakdown({}) == {}


class TestReadTurns:
    def test_a_truncated_final_line_is_skipped(self, tmp_path) -> None:
        # That is what a crash looks like, and a crash is when the log
        # is needed.
        p = tmp_path / "turns.jsonl"
        p.write_text('{"latency_ms": 900}\n{"latency_m')
        assert len(read_turns(p)) == 1

    def test_a_missing_file_reads_as_empty(self, tmp_path) -> None:
        assert read_turns(tmp_path / "nope.jsonl") == []
