"""Deciding which machine serves a turn, and when it is allowed to change.

Three tiers: `local` (the laptop Yash is sitting at), `lan` (the 3090 Ti
over ethernet, at home) and `tunnel` (the same box via Cloudflare, from
the hostel). Only the LLM and TTS ever move; `Locality.allows()` is what
enforces that, and it has a full truth-table test.

**Changes apply only at turn boundaries.** Never mid-turn. A turn that
began on the box and finished on the laptop would mix two models'
prosody inside one sentence, and its latency measurement would describe
neither tier. `Turn.tier` is snapshotted at creation for exactly this
reason.

**Asymmetric hysteresis, and the direction is deliberate.** Three
consecutive successes to promote, one failure to demote. Promotion is an
optimisation — being slow to take it costs a little latency. Demotion is
a failure recovery: being slow to take it means turns that produce
nothing at all. The expensive mistake is not symmetric, so neither is
the rule. It is the same principle as the speech gate and the expression
blender: leave fast, arrive slowly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from neiro.state import Locality, Tier

SUCCESSES_TO_PROMOTE = 3
FAILURES_TO_DEMOTE = 1
PROBE_INTERVAL_S = 2.0

# Preference order. `local` is last because it is the fallback that
# always works, not because it is worst.
PREFERENCE: tuple[Tier, ...] = (Tier.LAN, Tier.TUNNEL, Tier.LOCAL)


@dataclass
class TierResolver:
    """Which tier the next turn should use.

    `probe` is injected — reachability is a network question, and this
    class is about the decision, not the measurement.
    """

    probe: Callable[[Tier], bool] | None = None
    clock: Callable[[], float] = time.monotonic
    current: Tier = Tier.LOCAL
    interval_s: float = PROBE_INTERVAL_S
    _streak: dict[Tier, int] = field(default_factory=dict)
    _last_probe: float = -1e9

    def __post_init__(self) -> None:
        for tier in Tier:
            self._streak.setdefault(tier, 0)

    def observe(self, tier: Tier, reachable: bool) -> None:
        """Record one probe result. Does not change the tier by itself."""
        if reachable:
            self._streak[tier] = max(0, self._streak[tier]) + 1
        else:
            self._streak[tier] = min(0, self._streak[tier]) - 1

    def record_failure(self, tier: Tier) -> None:
        """A turn actually failed on this tier. Stronger evidence than a
        probe, because it is the thing we care about rather than a proxy
        for it — so it demotes immediately.
        """
        self._streak[tier] = -FAILURES_TO_DEMOTE

    def poll(self) -> None:
        """Probe every tier whose turn it is. Cheap and rate-limited."""
        if self.probe is None:
            return
        now = self.clock()
        if now - self._last_probe < self.interval_s:
            return
        self._last_probe = now
        for tier in PREFERENCE:
            if tier is Tier.LOCAL:
                continue  # the machine in front of him is always reachable
            try:
                self.observe(tier, bool(self.probe(tier)))
            except Exception:  # noqa: BLE001 — an unreachable tier is the answer, not an error
                self.observe(tier, False)

    def resolve(self) -> Tier:
        """The tier for the NEXT turn. Call at a turn boundary only.

        Promotion needs a sustained streak; demotion needs one failure.
        `local` is always available, so this cannot return nothing.
        """
        for tier in PREFERENCE:
            if tier is Tier.LOCAL:
                break
            if self._streak[tier] >= SUCCESSES_TO_PROMOTE:
                self.current = tier
                return tier
            if tier is self.current and self._streak[tier] > -FAILURES_TO_DEMOTE:
                # Already here and not yet failing — stay, rather than
                # flapping while the streak rebuilds.
                return tier
        self.current = Tier.LOCAL
        return Tier.LOCAL

    @staticmethod
    def allowed(locality: Locality, tier: Tier) -> bool:
        """The rule lives on the enum so there is one copy of it."""
        return locality.allows(tier)

    def tier_for(self, locality: Locality) -> Tier:
        """Where a provider with this locality should run *this* turn.

        A `LOCAL_PINNED` provider stays local even when the box is up —
        the mic and the desktop are wherever Yash is, and no amount of
        available GPU changes that.
        """
        chosen = self.current
        if locality.allows(chosen):
            return chosen
        for fallback in (Tier.LAN, Tier.LOCAL):
            if locality.allows(fallback):
                return fallback
        return Tier.LOCAL
