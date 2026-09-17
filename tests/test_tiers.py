"""Choosing a tier, and the asymmetry in when it may change."""

from __future__ import annotations

import pytest

from elizabeth.state import Locality, Tier
from elizabeth.tools.tiers import Reach
from elizabeth.tiers import FAILURES_TO_DEMOTE, SUCCESSES_TO_PROMOTE, TierResolver


class TestHysteresis:
    def test_it_starts_local(self) -> None:
        # The fallback that always works.
        assert TierResolver().resolve() is Tier.LOCAL

    def test_promotion_needs_a_sustained_streak(self) -> None:
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE - 1):
            r.observe(Tier.LAN, True)
            assert r.resolve() is Tier.LOCAL
        r.observe(Tier.LAN, True)
        assert r.resolve() is Tier.LAN

    def test_one_failure_demotes(self) -> None:
        # Promotion is an optimisation; demotion is failure recovery.
        # Being slow to demote means turns that produce nothing at all.
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE):
            r.observe(Tier.LAN, True)
        assert r.resolve() is Tier.LAN
        r.observe(Tier.LAN, False)
        assert r.resolve() is Tier.LOCAL

    def test_the_asymmetry_points_the_right_way(self) -> None:
        assert SUCCESSES_TO_PROMOTE > FAILURES_TO_DEMOTE

    def test_a_real_turn_failure_demotes_immediately(self) -> None:
        # Stronger evidence than a probe: it is the thing we care about
        # rather than a proxy for it.
        r = TierResolver()
        for _ in range(10):
            r.observe(Tier.LAN, True)
        assert r.resolve() is Tier.LAN
        r.record_failure(Tier.LAN)
        assert r.resolve() is Tier.LOCAL

    def test_it_does_not_flap_while_already_on_a_working_tier(self) -> None:
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE):
            r.observe(Tier.LAN, True)
        assert r.resolve() is Tier.LAN
        assert r.resolve() is Tier.LAN


class TestProbing:
    def test_probing_is_rate_limited(self) -> None:
        calls: list[Tier] = []
        now = [0.0]
        r = TierResolver(probe=lambda t: calls.append(t) or True, clock=lambda: now[0])
        r.poll()
        first = len(calls)
        r.poll()
        assert len(calls) == first, "a second poll within the interval must not re-probe"
        now[0] += r.interval_s + 0.1
        r.poll()
        assert len(calls) > first

    def test_local_is_never_probed(self) -> None:
        # The machine in front of him is always reachable by definition.
        calls: list[Tier] = []
        r = TierResolver(probe=lambda t: calls.append(t) or True)
        r.poll()
        assert Tier.LOCAL not in calls

    def test_a_probe_that_raises_counts_as_unreachable(self) -> None:
        # An unreachable tier is the answer, not an error.
        def boom(tier: Tier) -> bool:
            raise ConnectionError

        r = TierResolver(probe=boom)
        r.poll()
        assert r.resolve() is Tier.LOCAL

    def test_no_probe_at_all_is_fine(self) -> None:
        r = TierResolver(probe=None)
        r.poll()
        assert r.resolve() is Tier.LOCAL


class TestLocalityRouting:
    def test_a_pinned_provider_stays_local_even_when_the_box_is_up(self) -> None:
        # The mic and the desktop are wherever Yash is. No amount of
        # available GPU changes that.
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE):
            r.observe(Tier.LAN, True)
        r.resolve()
        assert r.tier_for(Locality.LOCAL_PINNED) is Tier.LOCAL

    def test_a_tierable_provider_follows_the_current_tier(self) -> None:
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE):
            r.observe(Tier.LAN, True)
        r.resolve()
        assert r.tier_for(Locality.TIERABLE) is Tier.LAN

    def test_stt_may_cross_the_lan_but_never_the_tunnel(self) -> None:
        # Shipping 16 kHz audio is fine over ethernet and breaks the
        # sub-second design over a tunnel.
        r = TierResolver()
        r.current = Tier.LAN
        assert r.tier_for(Locality.LAN_TIERABLE) is Tier.LAN
        r.current = Tier.TUNNEL
        assert r.tier_for(Locality.LAN_TIERABLE) is not Tier.TUNNEL

    @staticmethod
    def _at_the_hostel() -> TierResolver:
        # The tunnel is only ever current because the LAN failed to
        # build a streak: every probe of the ethernet address fails
        # while every probe of the tunnel succeeds.
        r = TierResolver()
        for _ in range(SUCCESSES_TO_PROMOTE):
            r.observe(Tier.LAN, False)
            r.observe(Tier.TUNNEL, True)
        assert r.resolve() is Tier.TUNNEL
        return r

    def test_stt_stays_local_when_the_tunnel_is_up_because_the_lan_is_down(self) -> None:
        # The resolver holds the evidence that the LAN is dead. Sending
        # STT there anyway means every utterance goes to an address
        # that cannot answer — she hears nothing while looking healthy.
        r = self._at_the_hostel()
        assert r.tier_for(Locality.LAN_TIERABLE) is Tier.LOCAL

    def test_stt_uses_a_lan_that_is_back_before_the_resolver_promotes_to_it(self) -> None:
        # One good probe is below the promotion streak, so the tunnel
        # stays current — but the LAN is reachable, and STT may use it.
        r = self._at_the_hostel()
        r.observe(Tier.LAN, True)
        assert r.resolve() is Tier.TUNNEL
        assert r.tier_for(Locality.LAN_TIERABLE) is Tier.LAN

    def test_a_turn_failure_on_the_lan_pulls_stt_back_to_local(self) -> None:
        # The route must self-correct from the strongest evidence there
        # is: a turn that actually failed there.
        r = self._at_the_hostel()
        r.observe(Tier.LAN, True)
        assert r.tier_for(Locality.LAN_TIERABLE) is Tier.LAN
        r.record_failure(Tier.LAN)
        assert r.tier_for(Locality.LAN_TIERABLE) is Tier.LOCAL
        assert r.resolve() is Tier.TUNNEL, "the tunnel did not fail; only the LAN did"

    @pytest.mark.parametrize("locality", list(Locality))
    def test_every_locality_resolves_to_something_it_allows(self, locality: Locality) -> None:
        r = TierResolver()
        for tier in Tier:
            r.current = tier
            assert locality.allows(r.tier_for(locality)), (locality, tier)

    def test_the_rule_has_one_copy(self) -> None:
        # On the enum, not duplicated here.
        for locality in Locality:
            for tier in Tier:
                assert TierResolver.allowed(locality, tier) == locality.allows(tier)


class TestReach:
    """`Reach` answers "whose network does this cross?", which is a
    different question from `tools.tiers.Tier`'s "how bad if it goes
    wrong?". docs/PRIVACY.md depends on the distinction holding.
    """

    def test_local_reach_runs_on_every_tier(self) -> None:
        # The overwhelming majority of tools. If this ever tightens,
        # every GREEN tool stops working away from home for no reason.
        for tier in Tier:
            assert Reach.LOCAL.allows(tier), tier

    def test_off_machine_reach_is_refused_only_on_the_tunnel(self) -> None:
        for reach in (Reach.OWN_MACHINES, Reach.INTERNET):
            assert reach.allows(Tier.LOCAL), reach
            assert reach.allows(Tier.LAN), reach
            assert not reach.allows(Tier.TUNNEL), reach

    def test_the_policy_can_only_ever_refuse_more_than_local_reach(self) -> None:
        # The asymmetry is the safety property: adding `reach` to a tool
        # must never let it run somewhere it could not run before.
        for reach in Reach:
            for tier in Tier:
                if reach is Reach.LOCAL:
                    continue
                assert reach.allows(tier) <= Reach.LOCAL.allows(tier), (reach, tier)
