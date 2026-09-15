"""Lane A: prosody z-scores → a ternary arousal band → `UserAffect`.

The whole differentiator lands here, and the design is mostly about what
it *refuses* to claim.

**Arousal is measured; valence is reported separately and distrusted.**
How activated someone sounds is audible — loud, high, fast, few pauses.
Whether they feel good or bad mostly is not: the best published system
on natural speech reaches macro-F1 0.43 across eight classes, and no
published number exists for Indian-accented English at all. So arousal
is a weighted composite of five features, and valence gets its own much
lower confidence and its own gate (G3b measures both, and the plan says
to assume neither).

**Ternary, not continuous.** The output that reaches the prompt is
"lower than usual / usual / higher than usual", because that is the
resolution the measurement actually supports. A continuous number would
imply a precision the signal does not have.

**Omitted, never softened.** Below the dead-band or the confidence
floor, `UserAffect.NONE` is returned and the annotation vanishes from
the prompt entirely. Telling her "he sounds normal" every turn is noise
she will eventually act on.

**Hysteresis.** A band change requires two of the last three windows to
agree. Without it the face flickers between states on a single
mispitched syllable, which reads as broken rather than alive.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from elizabeth.affect import features as feat
from elizabeth.affect.baseline import BaselineStore, SpeakerBaseline
from elizabeth.config import Elizabeth
from elizabeth.state import Locality, UserAffect

# How each feature contributes to arousal. Signs are the established
# direction: louder, higher, more variable, faster, and less pausing all
# read as more activated. Magnitudes say which are trusted most — energy
# and pace are the robust ones; pitch varies more with content than with
# state, and pausing is noisy on short windows.
AROUSAL_WEIGHTS: dict[str, float] = {
    "rms_mean": 0.30,
    "rms_p95": 0.15,
    "f0_median_hz": 0.20,
    "onset_rate_hz": 0.25,
    "pause_ratio": -0.10,
}

# Valence is a much weaker signal and is kept deliberately thin: wide
# pitch movement over a steady, unhurried baseline reads as positive
# more often than not. This exists to be MEASURED at gate G3b, not to be
# trusted before it. See docs/DECISIONS.md.
VALENCE_WEIGHTS: dict[str, float] = {
    "f0_iqr_hz": 0.6,
    "pause_ratio": -0.2,
    "voiced_ratio": 0.2,
}

BANDS = ("lower", "usual", "higher")
HYSTERESIS_WINDOW = 3
HYSTERESIS_AGREE = 2

# Valence never gets the same confidence as arousal from the same audio.
# One number, in one place, so the asymmetry is a decision rather than an
# accident of the maths.
VALENCE_CONFIDENCE_FACTOR = 0.5


def _composite(z_scores: dict[str, float], weights: dict[str, float]) -> float | None:
    """Weighted sum over whichever features were measurable.

    Renormalised by the weight actually present, so a window missing one
    feature is scaled correctly instead of silently reading low.
    """
    used = {k: w for k, w in weights.items() if k in z_scores}
    total = sum(abs(w) for w in used.values())
    if total <= 0:
        return None
    return sum(z_scores[k] * w for k, w in used.items()) / total


def band_for(z: float, dead_band: float) -> str:
    if z >= dead_band:
        return "higher"
    if z <= -dead_band:
        return "lower"
    return "usual"


def describe(affect: UserAffect, cfg: Elizabeth | None = None) -> str | None:
    """The `[voice: ...]` annotation, or `None` to omit it entirely.

    Words, not numbers: the model is being told how he sounded relative
    to his own normal, and "energy +1.4σ" invites it to do arithmetic it
    cannot do. Confidence is stated so the prompt's rule ("if
    tone-confidence is low, ignore it completely") has something to act
    on.
    """
    cfg = cfg or Elizabeth()
    if affect.confidence < cfg.affect.confidence_floor:
        return None
    band = band_for(affect.arousal_z, cfg.affect.dead_band_z)
    if band == "usual":
        return None
    energy = "more energy than usual" if band == "higher" else "flatter and quieter than usual"
    certainty = "high" if affect.confidence >= 0.7 else "medium"
    return f"{energy}, tone-confidence {certainty}"


@dataclass
class ProsodyAffectProvider:
    """protocols.AffectProvider. LOCAL_PINNED — the mic is wherever Yash is.

    Runs on a rolling window *while he is still speaking*, so it costs
    the turn budget nothing. `observe()` is called repeatedly during an
    utterance; `commit_utterance()` is called once at the end, and only
    that updates the baseline.
    """

    locality = Locality.LOCAL_PINNED

    cfg: Elizabeth = field(default_factory=Elizabeth)
    device: str = "default"
    store: BaselineStore = field(default_factory=BaselineStore.load)
    _recent_bands: deque[str] = field(default_factory=lambda: deque(maxlen=HYSTERESIS_WINDOW))
    _last_features: feat.ProsodyFeatures | None = None
    _band: str = "usual"

    @property
    def baseline(self) -> SpeakerBaseline:
        return self.store.for_device(self.device, self.cfg)

    def warm(self) -> float:
        """Pay librosa's JIT cost at process start — ~1.2 s, which is
        otherwise spent on the first thing he says.
        """
        return feat.warm(self.cfg)

    def _confidence(self, features: feat.ProsodyFeatures, n: int) -> float:
        """How much this reading should be believed, in [0, 1].

        Three things degrade it, all of them honest: a baseline that has
        not seen enough of him yet, a window that is mostly silence, and
        a window that is short. Below the warm-up count it is zero — not
        low, zero — because there is nothing to compare against.
        """
        if n < self.cfg.affect.warmup_utterances:
            return 0.0
        # sqrt, not linear. The z-score is only as good as the sigma
        # estimated from n samples, and the relative error of that
        # estimate shrinks as ~1/sqrt(2n) — roughly 32% at n=5, 16% at
        # n=20, 10% at n=50. A linear ramp made confidence far too
        # pessimistic early: with a 10-utterance baseline a +3.5σ reading
        # still scored 0.32, under the 0.45 floor, so she stayed silent
        # about tone for the first ~25 utterances of every new device
        # despite an unmistakable signal. sqrt puts n=5 (the documented
        # warm-up) right at the floor and rises from there.
        maturity = min(1.0, math.sqrt(n / max(1, self.cfg.affect.baseline_window)))
        voiced = min(1.0, features.voiced_ratio / 0.5)
        length = min(1.0, features.duration_s / self.cfg.affect.window_seconds)
        return float(max(0.0, min(1.0, 0.4 + 0.6 * maturity) * voiced * length))

    async def observe(self, pcm_window_16k: np.ndarray) -> UserAffect:
        features = feat.extract(pcm_window_16k, self.cfg)
        if features is None:
            return UserAffect.NONE
        self._last_features = features

        baseline = self.baseline
        z_scores = baseline.z_scores(features, self.cfg)
        if not z_scores:
            return UserAffect.NONE

        arousal = _composite(z_scores, AROUSAL_WEIGHTS)
        valence = _composite(z_scores, VALENCE_WEIGHTS)
        if arousal is None:
            return UserAffect.NONE

        # Hysteresis: the band only moves when two of the last three
        # windows agree, so one mispitched syllable can't flip her face.
        #
        # ...but only once there IS a history to disagree with. A short
        # utterance produces a single window, and requiring agreement
        # there meant every short sentence was penalised for
        # contradicting an empty deque — so a 2.3 s "what?!" could never
        # produce an annotation at all. Brevity is already accounted for
        # by the `length` term in `_confidence`; charging for it twice
        # silently disabled the feature on exactly the utterances most
        # likely to carry emotion.
        candidate = band_for(arousal, self.cfg.affect.dead_band_z)
        self._recent_bands.append(candidate)
        if (
            len(self._recent_bands) < HYSTERESIS_AGREE
            or self._recent_bands.count(candidate) >= HYSTERESIS_AGREE
        ):
            self._band = candidate
        confidence = self._confidence(features, baseline.n)
        if self._band != candidate:
            # Reported band and measured value disagree — the reading is
            # in transition, so say so by dropping confidence rather than
            # by reporting a number the band contradicts.
            confidence *= 0.5

        return UserAffect(
            arousal_z=float(arousal),
            valence_z=float(valence) if valence is not None else 0.0,
            confidence=confidence,
            baseline_n=baseline.n,
        )

    def valence_confidence(self, affect: UserAffect) -> float:
        """Valence is never as trustworthy as arousal from the same audio.

        Kept as an explicit method rather than folded into `observe()` so
        that anything reading valence has to ask for its confidence
        separately and cannot accidentally inherit arousal's.
        """
        return affect.confidence * VALENCE_CONFIDENCE_FACTOR

    def commit_utterance(self) -> bool:
        """End of turn: fold this utterance into his baseline. Returns
        True if drift detection reset it.

        Once per utterance, never per window — see `SpeakerBaseline.observe`.
        """
        if self._last_features is None:
            return False
        reset = self.baseline.observe(self._last_features, self.cfg)
        self._last_features = None
        self._recent_bands.clear()
        return reset

    def discard_utterance(self) -> None:
        """End of turn, but this utterance must not shape his normal:
        nothing usable was said, or it was cut off. The staged features
        and the hysteresis history both go — a rejected utterance's band
        must not decide the next one.
        """
        self._last_features = None
        self._recent_bands.clear()

    def save(self) -> None:
        self.store.save()
