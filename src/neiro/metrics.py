"""The one metric, and the per-turn record behind it.

**Milliseconds from the endpoint event to the first audio sample
reaching the sink.** That is the number. Not RTFx, not tokens/sec, not a
leaderboard figure — those are measured on batched workloads on
datacentre GPUs and predict nothing about one person saying one sentence
on a laptop. Rule (CLAUDE.md): never compare two options on any other
number.

Reported as p50 and p95 **separately**, never as a mean. Stage 2's
semantic turn model deliberately waits longer on an utterance that
sounds unfinished ("I want to go to... uh... the library") — that is the
feature working, and an average makes it look like a regression.

Every turn also appends a full stage-by-stage timeline to
~/.local/state/neiro/turns.jsonl, so when a turn feels slow the answer
is a row in a file rather than a guess about which stage moved.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from neiro.state import Turn

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "neiro"
TURNS_PATH = STATE_DIR / "turns.jsonl"

# The stage boundaries a turn passes through, in order. Used to render
# the waterfall — the point of which is that the next bottleneck names
# itself instead of being guessed at.
STAGE_ORDER = [
    "speech_start",
    "endpoint",
    "stt_done",
    "llm_first_token",
    "llm_first_sentence",
    "tts_first_chunk",
    "sink_played",
]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. No numpy dependency and no interpolation
    — with 20 samples, interpolating invents precision that isn't there.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # math.ceil, not round(x + 0.5): Python's round is banker's
    # rounding, so when p/100*n is an exact integer the index came out
    # by parity rather than by the nearest-rank definition. At n=10, p50
    # returned the 6th value where the definition says the 5th.
    index = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[index]


@dataclass
class TurnRecord:
    turn_id: int
    latency_ms: float | None
    stages_ms: dict[str, float]
    transcript: str | None = None
    reply: str | None = None
    emotion: str | None = None
    tier: str | None = None
    prompt_version: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "turn": self.turn_id,
                "latency_ms": self.latency_ms,
                "stages_ms": self.stages_ms,
                "transcript": self.transcript,
                "reply": self.reply,
                "emotion": self.emotion,
                "tier": self.tier,
                "prompt_version": self.prompt_version,
            }
        )


def headline_latency_ms(turn: Turn) -> float | None:
    """The one metric: endpoint -> first audio at the sink.

    None when the turn never got that far (rejected transcript, an
    error, a barge-in) — which is correct: a turn that produced no audio
    has no latency to report, and counting it as 0 would quietly
    improve every average it appears in.
    """
    start = turn.timeline.get("endpoint")
    end = turn.timeline.get("sink_played")
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def stage_durations_ms(turn: Turn) -> dict[str, float]:
    """Per-stage deltas, in order, for the waterfall."""
    present = [(name, turn.timeline[name]) for name in STAGE_ORDER if name in turn.timeline]
    durations = {}
    for (_, t0), (name, t1) in itertools.pairwise(present):
        durations[name] = (t1 - t0) * 1000.0
    return durations


def record_turn(turn: Turn, **extra: object) -> TurnRecord:
    """Build the record for one turn (does not write it)."""
    return TurnRecord(
        turn_id=turn.id,
        latency_ms=headline_latency_ms(turn),
        stages_ms=stage_durations_ms(turn),
        tier=turn.tier.value,
        **extra,  # type: ignore[arg-type]
    )


def append_turn(record: TurnRecord, path: Path | None = None) -> None:
    """Append one turn to the JSONL log, creating the directory if needed."""
    target = path or TURNS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(record.to_json() + "\n")


def summarise(latencies: list[float]) -> dict[str, float]:
    """p50 and p95, separately and deliberately — see module docstring."""
    return {
        "n": len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


def format_hud(record: TurnRecord) -> str:
    """One line per turn, printed live, so the cost of each stage is
    visible while using it rather than only in a later analysis.
    """
    parts = [f"turn {record.turn_id}"]
    for name, ms in record.stages_ms.items():
        parts.append(f"{name.replace('_', ' ')} {ms:.0f}ms")
    if record.latency_ms is not None:
        parts.append(f"[bold]E2E {record.latency_ms:.0f}ms[/bold]")
    else:
        parts.append("[dim]no audio[/dim]")
    return "  |  ".join(parts)
