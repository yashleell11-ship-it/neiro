"""Tests for the one metric.

The behaviour worth protecting here is mostly about honesty: a turn that
produced no audio must not be counted as fast, and p50/p95 must stay
separate from any notion of an average.
"""

from __future__ import annotations

import json

import pytest

from neiro.metrics import (
    TurnRecord,
    append_turn,
    format_hud,
    headline_latency_ms,
    percentile,
    record_turn,
    stage_durations_ms,
    summarise,
)
from neiro.state import Turn


def _turn_with(timeline: dict[str, float], turn_id: int = 1) -> Turn:
    turn = Turn.new(turn_id)
    turn.timeline.update(timeline)
    return turn


class TestHeadlineLatency:
    def test_endpoint_to_first_audio(self) -> None:
        turn = _turn_with({"endpoint": 10.0, "sink_played": 10.8})
        assert headline_latency_ms(turn) == pytest.approx(800.0)

    def test_turn_with_no_audio_has_no_latency(self) -> None:
        # A rejected transcript, an error, or a barge-in. Must be None,
        # not 0.0 — counting it as zero would quietly improve every
        # aggregate it appears in.
        turn = _turn_with({"endpoint": 10.0, "stt_done": 10.3})
        assert headline_latency_ms(turn) is None

    def test_turn_that_never_endpointed_has_no_latency(self) -> None:
        turn = _turn_with({"sink_played": 10.8})
        assert headline_latency_ms(turn) is None

    def test_empty_timeline(self) -> None:
        assert headline_latency_ms(Turn.new()) is None


class TestStageDurations:
    def test_consecutive_stages_become_deltas(self) -> None:
        turn = _turn_with(
            {
                "endpoint": 1.0,
                "stt_done": 1.2,
                "llm_first_token": 1.35,
                "sink_played": 1.6,
            }
        )
        stages = stage_durations_ms(turn)
        assert stages["stt_done"] == pytest.approx(200.0)
        assert stages["llm_first_token"] == pytest.approx(150.0)
        assert stages["sink_played"] == pytest.approx(250.0)

    def test_missing_stages_are_skipped_not_faked(self) -> None:
        turn = _turn_with({"endpoint": 1.0, "sink_played": 1.5})
        stages = stage_durations_ms(turn)
        assert list(stages) == ["sink_played"]
        assert stages["sink_played"] == pytest.approx(500.0)

    def test_single_stage_has_no_deltas(self) -> None:
        assert stage_durations_ms(_turn_with({"endpoint": 1.0})) == {}


class TestPercentile:
    def test_p50_of_odd_count(self) -> None:
        assert percentile([10, 20, 30], 50) == 20

    def test_p95_picks_the_tail(self) -> None:
        values = list(range(1, 21))  # 1..20
        assert percentile(values, 95) >= 19

    def test_p50_is_not_the_mean(self) -> None:
        # One pathological outlier must not drag the reported number —
        # which is exactly why p50 is reported rather than an average.
        values = [100, 100, 100, 100, 10000]
        assert percentile(values, 50) == 100
        mean = sum(values) / len(values)
        assert mean > 2000

    def test_empty_is_zero(self) -> None:
        assert percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert percentile([42], 50) == 42
        assert percentile([42], 95) == 42


class TestSummarise:
    def test_reports_p50_and_p95_separately(self) -> None:
        result = summarise([100.0, 200.0, 300.0, 400.0])
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "mean_ms" not in result  # deliberately absent
        assert result["n"] == 4

    def test_empty(self) -> None:
        assert summarise([])["n"] == 0


class TestRecordAndWrite:
    def test_record_carries_the_tier(self) -> None:
        turn = _turn_with({"endpoint": 1.0, "sink_played": 1.5}, turn_id=7)
        record = record_turn(turn)
        assert record.turn_id == 7
        assert record.tier == "laptop"
        assert record.latency_ms == pytest.approx(500.0)

    def test_append_writes_one_json_line(self, tmp_path) -> None:
        path = tmp_path / "turns.jsonl"
        turn = _turn_with({"endpoint": 1.0, "sink_played": 1.5})
        append_turn(record_turn(turn, transcript="hello"), path=path)
        append_turn(record_turn(turn, transcript="again"), path=path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["transcript"] == "hello"

    def test_append_creates_the_directory(self, tmp_path) -> None:
        path = tmp_path / "nested" / "deeper" / "turns.jsonl"
        append_turn(record_turn(_turn_with({})), path=path)
        assert path.exists()


class TestHud:
    def test_shows_the_end_to_end_number(self) -> None:
        turn = _turn_with({"endpoint": 1.0, "stt_done": 1.2, "sink_played": 1.6})
        line = format_hud(record_turn(turn))
        assert "E2E" in line
        assert "600ms" in line

    def test_says_so_when_there_was_no_audio(self) -> None:
        turn = _turn_with({"endpoint": 1.0, "stt_done": 1.2})
        line = format_hud(record_turn(turn))
        assert "no audio" in line

    def test_record_with_no_stages_still_renders(self) -> None:
        assert "turn 0" in format_hud(TurnRecord(turn_id=0, latency_ms=None, stages_ms={}))
