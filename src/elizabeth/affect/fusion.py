"""Combining Lane A (prosody) and Lane B (a trained model) into one reading.

A gap analysis on 2026-09-13 found this was the one genuinely
uncovered piece of the emotional core: two lanes were designed, and
nothing said what happens when both exist and disagree.

**The unit problem, which is the whole difficulty.** Lane A produces a
*z-score*: "1.4 sigma louder and faster than Yash's own normal". Lane B
produces a point on the circumplex: "arousal 0.6 in [-1, 1]", learned
from actors who are not Yash. Those are not the same quantity, and
averaging them is a category error that would look like it worked.

The fix is to baseline Lane B too. Run the model over his ordinary
utterances, keep a rolling robust estimate of *its predictions on him*,
and z-score its output the same way. Then both lanes answer the same
question — "how far from his usual is this?" — and can be combined.
That also quietly fixes the model's calibration: a SER model that reads
every Indian-accented voice as slightly angry has a non-zero median on
him, and subtracting that median removes the bias entirely.

**Disagreement is information, not noise.** When the lanes point
opposite ways, the answer is not their average — it is lower
confidence. Two independent measurements that contradict each other
mean the reading is unreliable, and the prompt's rule ("if
tone-confidence is low, ignore it completely") is what acts on that.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from elizabeth.affect.baseline import _mad, _median
from elizabeth.config import Elizabeth
from elizabeth.state import UserAffect

# How much each lane is worth when both are confident and agree. Lane A
# is measured against his own voice from the first day; Lane B is a model
# trained mostly on actors. Until gate G3b says otherwise on HIS
# recordings, the lane that knows him personally leads.
LANE_A_WEIGHT = 0.6
LANE_B_WEIGHT = 0.4

# Confidence multiplier when the lanes point opposite ways. Not zero:
# one lane being wrong is likelier than both being useless, so a
# contradicted reading is weak rather than absent.
DISAGREEMENT_PENALTY = 0.4

# Confidence multiplier when they agree on direction. Independent
# agreement is real evidence and should read stronger than either alone,
# capped at 1.0.
AGREEMENT_BONUS = 1.25


@dataclass
class LaneBCalibration:
    """A rolling, robust z-scorer for one model's own predictions on Yash.

    Deliberately the same median/MAD machinery as `SpeakerBaseline`: the
    reason is identical (one shout must not redefine normal), and having
    two different notions of "his usual" in one system is how they drift
    apart.
    """

    window: int = 50
    arousal: deque[float] = field(default_factory=deque)
    valence: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.arousal = deque(self.arousal, maxlen=self.window)
        self.valence = deque(self.valence, maxlen=self.window)

    @property
    def n(self) -> int:
        return len(self.arousal)

    def observe(self, arousal: float, valence: float) -> None:
        self.arousal.append(float(arousal))
        self.valence.append(float(valence))

    def _z(self, value: float, samples: deque[float], cfg: Elizabeth) -> float | None:
        if not samples:
            return None
        values = list(samples)
        centre = _median(values)
        sigma = _mad(values, centre) * cfg.affect.mad_to_sigma
        # Lane B's outputs live in [-1, 1], so a fraction-of-median floor
        # collapses near zero — the scale floor has to be absolute here.
        sigma = max(sigma, cfg.affect.min_sigma_fraction)
        z = (value - centre) / sigma
        return max(-cfg.affect.max_abs_z, min(cfg.affect.max_abs_z, z))

    def to_z(
        self, arousal: float, valence: float, cfg: Elizabeth | None = None
    ) -> tuple[float | None, float | None]:
        cfg = cfg or Elizabeth()
        return self._z(arousal, self.arousal, cfg), self._z(valence, self.valence, cfg)


def _same_direction(a: float, b: float, dead_band: float) -> bool | None:
    """True both point the same way, False opposite, None one is neutral.

    A lane sitting inside the dead-band is not disagreeing with anything
    — it is declining to say — so it must not be counted as a conflict.
    """
    a_side = 0 if abs(a) < dead_band else (1 if a > 0 else -1)
    b_side = 0 if abs(b) < dead_band else (1 if b > 0 else -1)
    if a_side == 0 or b_side == 0:
        return None
    return a_side == b_side


def fuse(
    lane_a: UserAffect | None,
    lane_b: UserAffect | None,
    cfg: Elizabeth | None = None,
) -> UserAffect:
    """One reading from up to two lanes. Both `None` gives `UserAffect.NONE`.

    Weighting is by each lane's own confidence *and* its standing weight,
    so a lane that says "I am unsure" contributes proportionally little
    instead of dragging the answer halfway toward itself.
    """
    cfg = cfg or Elizabeth()
    lanes = [
        (lane_a, LANE_A_WEIGHT),
        (lane_b, LANE_B_WEIGHT),
    ]
    usable = [
        (affect, weight) for affect, weight in lanes if affect is not None and affect.confidence > 0
    ]
    if not usable:
        return UserAffect.NONE
    if len(usable) == 1:
        # One lane: pass it through unchanged. Inventing a confidence
        # penalty for "only one opinion" would punish the normal case,
        # which for most of this project's life is Lane A alone.
        return usable[0][0]

    total = sum(affect.confidence * weight for affect, weight in usable)
    if total <= 0:
        return UserAffect.NONE
    arousal = sum(a.arousal_z * a.confidence * w for a, w in usable) / total
    valence = sum(a.valence_z * a.confidence * w for a, w in usable) / total
    confidence = sum(a.confidence * w for a, w in usable) / sum(w for _, w in usable)

    agree = _same_direction(lane_a.arousal_z, lane_b.arousal_z, cfg.affect.dead_band_z)
    if agree is True:
        confidence = min(1.0, confidence * AGREEMENT_BONUS)
    elif agree is False:
        confidence *= DISAGREEMENT_PENALTY

    return UserAffect(
        arousal_z=float(arousal),
        valence_z=float(valence),
        confidence=float(max(0.0, min(1.0, confidence))),
        events=tuple(dict.fromkeys(lane_a.events + lane_b.events)),
        baseline_n=min(lane_a.baseline_n, lane_b.baseline_n),
    )
