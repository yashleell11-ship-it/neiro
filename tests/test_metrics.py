"""Tests for the one metric.

The behaviour worth protecting here is mostly about honesty: a turn that
produced no audio must not be counted as fast, and p50/p95 must stay
separate from any notion of an average.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from elizabeth.metrics import (
    STAGE_ORDER,
    TurnRecord,
    append_turn,
    format_hud,
    headline_latency_ms,
    percentile,
    record_turn,
    stage_durations_ms,
    summarise,
)
from elizabeth.orchestrator import Orchestrator
from elizabeth.state import Turn

REPO = Path(__file__).resolve().parents[1]

# Names the pipeline stamps that are deliberately NOT waterfall stages.
# Each one is here because it is a real `turn.stamp(...)` in the tree and
# the cross-check below would otherwise flag it as drift.
STAMPED_BUT_NOT_A_STAGE = {
    # Terminal outcomes of a turn that produced no audio. A bar for them
    # would put a duration on a turn that has none.
    "cancelled",
    "failed",
    # The file sink's stand-in for `sink_played`, named differently on
    # purpose so a file-sink number is never averaged with a browser one.
    "sink_written_seq0",
}


KNOWN_STAMPS = set(STAGE_ORDER) | STAMPED_BUT_NOT_A_STAGE


def _turn_with(timeline: dict[str, float], turn_id: int = 1) -> Turn:
    # A fixture may only use names the pipeline really stamps. This file
    # once built its timelines from names nothing stamped, and every test
    # in it passed while the waterfall was wrong for every real turn.
    unknown = set(timeline) - KNOWN_STAMPS
    assert not unknown, f"not a stamp the pipeline makes: {sorted(unknown)}"
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
                "tts_first_chunk": 1.35,
                "sink_played": 1.6,
            }
        )
        stages = stage_durations_ms(turn)
        assert stages["stt_done"] == pytest.approx(200.0)
        assert stages["tts_first_chunk"] == pytest.approx(150.0)
        assert stages["sink_played"] == pytest.approx(250.0)

    def test_missing_stages_are_skipped_not_faked(self) -> None:
        turn = _turn_with({"endpoint": 1.0, "sink_played": 1.5})
        stages = stage_durations_ms(turn)
        assert list(stages) == ["sink_played"]
        assert stages["sink_played"] == pytest.approx(500.0)

    def test_single_stage_has_no_deltas(self) -> None:
        assert stage_durations_ms(_turn_with({"endpoint": 1.0})) == {}


def _names_stamped_in_the_tree() -> set[str]:
    """Every name passed to a `.stamp(...)` call under src/ and scripts/.

    Read from the AST, not with a regex: a comment that mentions
    `turn.stamp(...)` is prose, not a call site, and a regex cannot tell
    the two apart.

    scripts/ is included because the bench's fake browser sink is, today,
    the only thing that stamps `sink_played` — the browser sink that
    will stamp it for real is not built yet.
    """
    names: set[str] = set()
    for folder in ("src", "scripts"):
        for path in sorted((REPO / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "stamp":
                    continue
                arg = node.args[0] if node.args else None
                # A stamp whose name is a variable would slip past this
                # scan unseen, and the scan is the whole guarantee. Refuse.
                assert isinstance(arg, ast.Constant) and isinstance(arg.value, str), (
                    path,
                    node.lineno,
                )
                names.add(arg.value)
    return names


class TestStageOrderFollowsThePipeline:
    """Regression: STAGE_ORDER was a stale copy of the stage contract.

    It was written before the orchestrator existed and never reconciled
    with it. Three of its names were stamped by nothing, four boundaries
    the pipeline does stamp were absent, and nothing failed — the
    waterfall just quietly folded the emotion resolve and the whole LLM
    stream into a bar labelled `tts_first_chunk`. The tests in this file
    could not catch it because they fabricated timelines from the same
    phantom names.
    """

    def test_every_stage_is_stamped_and_every_stamp_is_accounted_for(self) -> None:
        stamped = _names_stamped_in_the_tree()
        assert stamped, "the scan found no stamps at all — is REPO right?"
        # No phantoms: a stage nothing stamps can never render a bar.
        assert set(STAGE_ORDER) <= stamped, set(STAGE_ORDER) - stamped
        # No strays: a stamp the waterfall does not know is a boundary it
        # cannot show. It is either a stage or on the documented list.
        assert stamped - set(STAGE_ORDER) == STAMPED_BUT_NOT_A_STAGE

    def test_the_exception_list_holds_only_real_stamps(self) -> None:
        # Otherwise a stamp could be deleted from the pipeline and the
        # exception list would keep vouching for it forever.
        assert STAMPED_BUT_NOT_A_STAGE <= _names_stamped_in_the_tree()
        assert not STAMPED_BUT_NOT_A_STAGE & set(STAGE_ORDER)

    def test_a_real_turn_stamps_exactly_the_waterfall_in_its_order(self) -> None:
        # The seam the fabricated fixtures skipped: the orchestrator's own
        # stamps, read back through metrics.py. Every stage present,
        # nothing unknown, the order the waterfall assumes, one bar per
        # boundary, and a headline number at the end of it.
        orch = Orchestrator(stt=_Stt(), llm=_Llm(), tts=_Tts(), sink=_BrowserShapedSink())
        result = asyncio.run(orch.run(np.zeros(16000, dtype=np.float32)))
        assert result.error is None and not result.cancelled

        timeline = result.turn.timeline
        assert set(timeline) == set(STAGE_ORDER)
        assert sorted(timeline, key=timeline.__getitem__) == list(STAGE_ORDER)
        assert list(stage_durations_ms(result.turn)) == list(STAGE_ORDER[1:])
        headline = headline_latency_ms(result.turn)
        assert headline is not None and headline > 0


class _Stt:
    async def transcribe(self, pcm: np.ndarray) -> str:
        return "what's my battery at"


class _Llm:
    reply = "<e:happy:7> Ninety six percent. Still charging."

    async def stream(self, messages: list[dict], tools=None):
        for i in range(0, len(self.reply), 5):
            yield {"text": self.reply[i : i + 5]}


class _Tts:
    async def synth(self, text: str, state=None):
        yield np.zeros(240, dtype=np.float32), None


class _BrowserShapedSink:
    """Stamps `sink_played` the way the browser will: from a later
    message, after `play()` has returned and `sink_first_sent` is down.

    Stamping inside `play()` — what the bench's fake does — lands
    `sink_played` BEFORE `sink_first_sent`. That is the file sink's
    shape, not the browser's, and the order test above would then pass
    or fail on a fake's accident rather than on the pipeline.
    """

    async def play(self, turn: Turn, pcm, seq: int = 0, text: str = "", visemes=None) -> None:
        if seq == 0:
            asyncio.get_running_loop().call_soon(turn.stamp, "sink_played")

    async def cancel(self, turn: Turn) -> None:
        pass


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
        assert record.tier == "local"
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
