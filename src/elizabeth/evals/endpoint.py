"""Scoring the endpointer: does she wait long enough, and not too long?

Two bars, measured separately, because they fail in opposite directions
and averaging them hides both.

**Did she cut him off?** A false cut ends a turn mid-sentence. It costs
the whole utterance and the goodwill with it, so it is reported as its
own rate rather than folded into an accuracy.

**How long did she make him wait?** Acceptance is on **p50**, with p90
reported separately. A mean would let one 3-second hang disappear behind
nineteen fast turns, and the 3-second hang is the one that feels broken.

The hesitation fixture is the point of the whole component: *"I want to
go to… uh… the library"* must stay one turn. It is scored on its own,
because a system that handles ordinary speech perfectly and cuts every
hesitation is worse than one slightly slower at both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# From the plan's Stage 2 budget: Silero trigger + smart-turn decision.
TARGET_P50_MS = 235.0
MAX_FALSE_CUT_RATE = 0.02  # 2% — a cut sentence is expensive


@dataclass(frozen=True)
class EndpointScore:
    n: int
    p50_ms: float
    p90_ms: float
    false_cut_rate: float
    missed_rate: float
    hesitations_survived: float | None

    @property
    def passed(self) -> bool:
        # Both bars, never averaged: they fail in opposite directions.
        return self.p50_ms <= TARGET_P50_MS and self.false_cut_rate <= MAX_FALSE_CUT_RATE

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "p50_ms": round(self.p50_ms, 1),
            "p90_ms": round(self.p90_ms, 1),
            "false_cut_rate": round(self.false_cut_rate, 4),
            "missed_rate": round(self.missed_rate, 4),
            "hesitations_survived": (
                round(self.hesitations_survived, 4)
                if self.hesitations_survived is not None
                else None
            ),
            "target_p50_ms": TARGET_P50_MS,
            "max_false_cut_rate": MAX_FALSE_CUT_RATE,
            "verdict": "GO" if self.passed else "NO-GO",
        }


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank, matching evals/latency.py — one definition of a
    percentile across the project, so two numbers are comparable.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def score(
    decisions: list[tuple[bool, bool, float]],
    hesitations: list[bool] | None = None,
) -> EndpointScore:
    """`decisions` is `(fired, was_really_the_end, delay_ms)` per candidate.

    `hesitations` is a separate list of "did this pause-containing
    utterance survive as one turn" — scored apart because handling
    ordinary speech well while cutting every hesitation is a worse system
    than being slightly slower at both.
    """
    ends = [(fired, delay) for fired, real, delay in decisions if real]
    non_ends = [fired for fired, real, _ in decisions if not real]

    delays = [d for fired, d in ends if fired]
    false_cuts = sum(1 for f in non_ends if f)
    missed = sum(1 for fired, _ in ends if not fired)

    return EndpointScore(
        n=len(decisions),
        p50_ms=percentile(delays, 0.50),
        p90_ms=percentile(delays, 0.90),
        false_cut_rate=(false_cuts / len(non_ends)) if non_ends else 0.0,
        missed_rate=(missed / len(ends)) if ends else 0.0,
        hesitations_survived=(sum(hesitations) / len(hesitations)) if hesitations else None,
    )
