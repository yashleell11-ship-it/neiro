"""Smoothing her expression so the face reads as alive, not as a mask.

A VRM's `expressionManager` takes a weight per preset. Setting those
weights directly from each new `ElizabethState` makes the face *snap*
between expressions, which reads as a puppet. What reads as alive is an
exponential approach with **asymmetric time constants** — expressions
arrive faster than they leave, like real faces do.

    weight += (target - weight) * (1 - exp(-dt / tau))

`tau` is 0.12 s rising and 0.30 s falling. That asymmetry is the single
most important number here; making them equal is what makes an avatar
look mechanical even when everything else is right.

Three further rules, each earned:

**Surprise decays on its own.** It is physiologically brief. Held past
about a second it stops reading as surprise and starts reading as a
stare — so it is released after `surprised_hold_s` whether or not a new
state has arrived. The research pass also found `surprised` is often
*unbound* in real VRM models, so `available` filters targets to what a
given avatar actually has, and the weight is redistributed rather than
silently dropped.

**Weights are capped in total.** VRM expressions are additive on one
mesh; a sum past 1.0 gives geometry that looks broken. Scaled down
together so the *mix* survives the clamp.

**Tiny weights are snapped to zero.** An exponential never quite
arrives, and 0.003 of "angry" left on her face all evening is a real
thing that happens.

This runs on the Python side, not in the browser, so the same blend can
be unit-tested and so `state{}` frames on the WebSocket carry already-
smoothed weights — the browser animates the mouth on the audio clock and
nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from elizabeth.config import Elizabeth
from elizabeth.state import ElizabethState, EmotionLabel

# The five expressive presets. `neutral` is deliberately excluded: in
# VRM it is the rest pose, and driving it as a weight fights the others
# instead of blending with them. Neutral is "all of these near zero".
EXPRESSIVE: tuple[EmotionLabel, ...] = (
    EmotionLabel.HAPPY,
    EmotionLabel.ANGRY,
    EmotionLabel.SAD,
    EmotionLabel.RELAXED,
    EmotionLabel.SURPRISED,
)


@dataclass
class ExpressionBlender:
    """Turns a stream of `ElizabethState` into a stream of VRM weight maps.

    `available` is what the loaded avatar actually reports it can do —
    the browser sends it in `ready{expressions[]}`. A target the model
    lacks is redistributed to the nearest one it has, never silently
    dropped, because a face that does nothing is indistinguishable from a
    bug.
    """

    cfg: Elizabeth = field(default_factory=Elizabeth)
    available: frozenset[str] | None = None
    weights: dict[str, float] = field(default_factory=dict)
    _surprised_age: float = 0.0

    def __post_init__(self) -> None:
        for label in EXPRESSIVE:
            self.weights.setdefault(label.value, 0.0)

    # Where a target goes when the avatar cannot do it. Chosen by
    # closeness on the circumplex, so the face still moves the right way
    # even on a model missing presets.
    FALLBACKS: ClassVar[dict[EmotionLabel, tuple[EmotionLabel, ...]]] = {
        EmotionLabel.SURPRISED: (EmotionLabel.HAPPY, EmotionLabel.ANGRY),
        EmotionLabel.RELAXED: (EmotionLabel.HAPPY,),
        EmotionLabel.ANGRY: (EmotionLabel.SAD,),
        EmotionLabel.HAPPY: (EmotionLabel.RELAXED,),
        EmotionLabel.SAD: (EmotionLabel.RELAXED,),
    }

    def _resolve(self, label: EmotionLabel) -> str | None:
        """The preset to actually drive for this label, or None if the
        avatar can express nothing close.
        """
        if self.available is None or label.value in self.available:
            return label.value
        for alternative in self.FALLBACKS.get(label, ()):
            if alternative.value in self.available:
                return alternative.value
        return None

    def target_weights(self, state: ElizabethState) -> dict[str, float]:
        """Where the face should end up for this state, before smoothing.

        One expression at a time, at the tag's own intensity. Mixing
        several is how you get a face that means nothing in particular.
        """
        targets = dict.fromkeys((label.value for label in EXPRESSIVE), 0.0)
        if state.label is EmotionLabel.NEUTRAL:
            return targets
        preset = self._resolve(state.label)
        if preset is not None:
            targets[preset] = max(0.0, min(1.0, state.intensity))
        return targets

    def step(self, state: ElizabethState, dt: float) -> dict[str, float]:
        """Advance the blend by `dt` seconds toward `state`.

        Called once per animation frame. `dt` is real elapsed time, not a
        fixed step, so a dropped frame does not slow the expression down.
        """
        if dt <= 0:
            return dict(self.weights)

        targets = self.target_weights(state)

        # Surprise releases itself, regardless of what the model is still
        # saying, once it has been held long enough.
        surprised = EmotionLabel.SURPRISED.value
        if targets.get(surprised, 0.0) > 0:
            self._surprised_age += dt
            if self._surprised_age > self.cfg.expression.surprised_hold_s:
                decay = self.cfg.expression.surprised_decay_s
                over = self._surprised_age - self.cfg.expression.surprised_hold_s
                targets[surprised] *= math.exp(-over / max(decay, 1e-6))
        else:
            self._surprised_age = 0.0

        rise, fall = self.cfg.expression.tau_rise_s, self.cfg.expression.tau_fall_s
        for name, current in list(self.weights.items()):
            target = targets.get(name, 0.0)
            tau = rise if target > current else fall
            alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
            value = current + (target - current) * alpha
            self.weights[name] = 0.0 if value < self.cfg.expression.epsilon else value

        self._clamp_total()
        return dict(self.weights)

    def _clamp_total(self) -> None:
        limit = self.cfg.expression.max_total_weight
        total = sum(self.weights.values())
        if total > limit and total > 0:
            # Scale together so the mix is preserved — zeroing the
            # smaller ones instead would make a crossfade jump.
            scale = limit / total
            for name in self.weights:
                self.weights[name] *= scale

    def settle(
        self, state: ElizabethState, seconds: float = 2.0, fps: float = 60.0
    ) -> dict[str, float]:
        """Run the blend forward to where it would end up. For tests and
        for the one-shot case where no animation loop is running.
        """
        dt = 1.0 / fps
        for _ in range(int(seconds * fps)):
            self.step(state, dt)
        return dict(self.weights)

    def dominant(self) -> tuple[str, float]:
        """The strongest expression right now, for the HUD and logs."""
        if not self.weights:
            return (EmotionLabel.NEUTRAL.value, 0.0)
        name, value = max(self.weights.items(), key=lambda kv: kv[1])
        return (
            (name, value)
            if value >= self.cfg.expression.epsilon
            else (EmotionLabel.NEUTRAL.value, 0.0)
        )
