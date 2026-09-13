"""The one metric, and the only honest way to compute it.

**Endpoint event → the browser's `played` callback for sequence 0.**
p50 and p95, never a mean, per tier and per device profile. Everything
before that callback is "we sent it"; only the browser knows when sound
actually left the speaker.

Three rules this file exists to enforce, because each is easy to break
by accident and none of them produce an error when broken:

**No means.** A mean hides the turn that took four seconds behind
nineteen that took eight hundred milliseconds, and the four-second turn
is the one that makes her feel broken. p50 says what it is usually like;
p95 says what it is like when it is bad. `summarise()` refuses to emit a
mean at all.

**Never compare two options on any number but this one.** Not RTFx, not
tokens/second, not a leaderboard score. A faster decoder behind a slower
first clause is a slower assistant.

**Tiers and device profiles never merge.** A p50 computed over a mix of
laptop and box turns describes neither, and the same is true of earbuds
against speakers.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# The stages a turn stamps, in order. `sink_played` is the browser's
# callback; `sink_written_seq0` is the file sink's nearest equivalent and
# is deliberately named differently so the two are never averaged.
STAGES = (
    "endpoint",
    "stt_start",
    "stt_done",
    "emotion_resolved",
    "tts_first_chunk",
    "sink_first_sent",
    "sink_played",
    "turn_done",
)
HEADLINE_END = "sink_played"


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation, no numpy.

    Interpolating invents a latency nobody experienced. With twenty
    turns the honest p95 is the second-worst one that happened.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True)
class LatencySummary:
    n: int
    p50_ms: float
    p95_ms: float
    worst_ms: float
    tier: str
    profile: str

    def as_dict(self) -> dict:
        # No mean. Deliberately absent, not merely unused.
        return {
            "n": self.n,
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "worst_ms": round(self.worst_ms, 1),
            "tier": self.tier,
            "profile": self.profile,
        }


def headline_ms(timeline: dict[str, float]) -> float | None:
    """Endpoint → first audio actually played, in ms.

    `None` when the turn produced no audio — a cancelled or failed turn
    has no latency, and recording it as 0.0 would drag every percentile
    toward a number that never happened.
    """
    start = timeline.get("endpoint")
    end = timeline.get(HEADLINE_END)
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def stage_breakdown(timeline: dict[str, float]) -> dict[str, float]:
    """Milliseconds spent in each consecutive stage that was stamped.

    The waterfall is what turns "it feels slow" into "STT is 140 ms and
    the LLM is fifteen seconds".
    """
    present = [s for s in STAGES if s in timeline]
    out: dict[str, float] = {}
    for earlier, later in itertools.pairwise(present):
        out[f"{earlier}->{later}"] = (timeline[later] - timeline[earlier]) * 1000.0
    return out


def summarise(
    records: Iterable[dict], tier: str = "local", profile: str = "earbuds"
) -> LatencySummary:
    """Percentiles over turns from ONE tier and ONE device profile.

    Filtering rather than grouping, on purpose: a caller that wants both
    must ask twice and will therefore report two numbers, instead of one
    that describes neither.
    """
    values = [
        ms
        for r in records
        if r.get("tier", tier) == tier and r.get("profile", profile) == profile
        for ms in (r.get("latency_ms") or headline_ms(r.get("timeline") or {}),)
        if ms is not None
    ]
    return LatencySummary(
        n=len(values),
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        worst_ms=max(values) if values else float("nan"),
        tier=tier,
        profile=profile,
    )


def read_turns(path: Path) -> list[dict]:
    """Load `turns.jsonl`. A truncated final line is skipped — that is
    what a crash looks like, and a crash is when the log is needed.
    """
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records
