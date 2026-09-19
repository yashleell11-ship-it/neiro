"""The typed tool registry.

The threat model is specific and was proven by experiment during the
planning pass: `hyprctl dispatch X` on this machine is a Lua 5.5 eval
with `os.execute` in scope. So the rule under test is not "validate the
model's strings carefully" — it is that **the model never emits a string
that reaches a command at all**.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from elizabeth.tools.audit import ATTEMPTED, DENIED, FAILED, RATE_LIMITED, SUCCEEDED, AuditLog
from elizabeth.state import Tier as StateTier
from elizabeth.tools.registry import (
    RateLimited,
    ReachRefused,
    SourceRefused,
    ToolNotConfirmed,
    ToolRegistry,
    ToolRejected,
    ToolSpec,
)
from elizabeth.tools.tiers import Reach, Source, Tier


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int


class EnumArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["up", "down"]
    steps: int = 1


class FreeTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str


class SloppyArgs(BaseModel):
    index: int  # no extra="forbid"


def spec(name="ping", tier=Tier.GREEN, model=NoArgs, handler=None, **kw) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        tier=tier,
        args_model=model,
        handler=handler or (lambda **_: "ok"),
        **kw,
    )


class TestRegistrationRefusesUnsafeShapes:
    def test_a_bare_string_argument_is_refused_at_registration(self) -> None:
        # At registration, not at call time: a tool with a free-form
        # string should stop the process on startup, not fail once in
        # production when a 4B model finally emits the wrong thing.
        with pytest.raises(ValueError, match="never emit a string"):
            ToolRegistry().register(spec(model=FreeTextArgs))

    def test_an_int_index_is_fine(self) -> None:
        assert ToolRegistry().register(spec(model=IndexArgs))

    def test_a_literal_enum_is_fine(self) -> None:
        assert ToolRegistry().register(spec(model=EnumArgs))

    def test_spoken_text_must_be_opted_into_explicitly(self) -> None:
        r = ToolRegistry()
        assert r.register(spec(model=FreeTextArgs, spoken_text_fields=frozenset({"target"})))

    def test_extra_forbid_is_mandatory(self) -> None:
        # An unexpected key from the model is a rejection, not something
        # to ignore.
        with pytest.raises(ValueError, match="extra='forbid'"):
            ToolRegistry().register(spec(model=SloppyArgs))

    def test_red_tools_cannot_be_registered_at_all(self) -> None:
        # Not gated — absent. The model cannot express what it cannot see.
        with pytest.raises(ValueError, match="must not be registered"):
            ToolRegistry().register(spec(name="rm", tier=Tier.RED))

    def test_duplicate_names_are_refused(self) -> None:
        r = ToolRegistry()
        r.register(spec())
        with pytest.raises(ValueError, match="duplicate"):
            r.register(spec())


class TestSchemas:
    def test_the_schema_comes_from_the_validating_model(self) -> None:
        # They cannot drift, which is the usual way a registry quietly
        # stops matching what it advertises.
        r = ToolRegistry()
        r.register(spec(name="brightness", tier=Tier.YELLOW, model=EnumArgs))
        fn = r.schemas()[0]["function"]
        assert fn["name"] == "brightness"
        props = fn["parameters"]["properties"]
        assert props["direction"]["enum"] == ["up", "down"]
        assert fn["parameters"]["additionalProperties"] is False

    def test_green_ceiling_hides_yellow_tools(self) -> None:
        # On a tier where confirmation is impossible, the model must not
        # even be offered the tool.
        r = ToolRegistry()
        r.register(spec(name="read", tier=Tier.GREEN))
        r.register(spec(name="write", tier=Tier.YELLOW, model=IndexArgs))
        assert {s["function"]["name"] for s in r.schemas(Tier.GREEN)} == {"read"}
        assert len(r.schemas()) == 2

    def test_order_is_stable(self) -> None:
        # The tools array is part of the KV prefix. Reordering it costs a
        # full prefill every turn, with no error to explain the latency.
        r = ToolRegistry()
        for n in ("zeta", "alpha", "mid"):
            r.register(spec(name=n))
        names = [s["function"]["name"] for s in r.schemas()]
        assert names == sorted(names)
        assert names == [s["function"]["name"] for s in r.schemas()]


class TestValidation:
    def test_an_unknown_tool_names_the_real_ones(self) -> None:
        # Turns a hallucinated call into a recoverable turn. Safe: the
        # model already has the list.
        r = ToolRegistry()
        r.register(spec(name="volume", model=IndexArgs))
        with pytest.raises(ToolRejected, match="volume"):
            r.call("volumee", {"index": 1})

    def test_an_unexpected_key_is_rejected(self) -> None:
        r = ToolRegistry()
        r.register(spec(model=IndexArgs))
        with pytest.raises(ToolRejected):
            r.call("ping", {"index": 1, "shell": "rm -rf /"})

    def test_a_wrong_type_is_rejected(self) -> None:
        r = ToolRegistry()
        r.register(spec(model=IndexArgs))
        with pytest.raises(ToolRejected):
            r.call("ping", {"index": "$(whoami)"})

    def test_an_out_of_enum_value_is_rejected(self) -> None:
        r = ToolRegistry()
        r.register(spec(model=EnumArgs))
        with pytest.raises(ToolRejected):
            r.call("ping", {"direction": "sideways"})

    def test_validate_does_not_run_the_handler(self) -> None:
        ran = []
        r = ToolRegistry()
        r.register(spec(model=IndexArgs, handler=lambda **_: ran.append(1) or "ok"))
        r.validate("ping", {"index": 0})
        assert not ran


class TestTierGating:
    def test_green_runs_without_confirmation(self) -> None:
        r = ToolRegistry()
        r.register(spec(handler=lambda **_: "battery 96 percent"))
        assert r.call("ping", {}) == "battery 96 percent"

    def test_yellow_without_a_confirmer_refuses(self) -> None:
        r = ToolRegistry()
        r.register(spec(tier=Tier.YELLOW, model=IndexArgs))
        with pytest.raises(ToolNotConfirmed):
            r.call("ping", {"index": 1})

    def test_yellow_runs_when_confirmed(self) -> None:
        r = ToolRegistry(confirm=lambda *_: True)
        r.register(spec(tier=Tier.YELLOW, model=IndexArgs, handler=lambda **_: "volume 30 percent"))
        assert r.call("ping", {"index": 3}) == "volume 30 percent"

    def test_yellow_stops_when_denied(self) -> None:
        ran = []
        r = ToolRegistry(confirm=lambda *_: False)
        r.register(
            spec(tier=Tier.YELLOW, model=IndexArgs, handler=lambda **_: ran.append(1) or "done")
        )
        with pytest.raises(ToolNotConfirmed):
            r.call("ping", {"index": 3})
        assert not ran, "a denied tool must not run"


class TestNonce:
    """The confirmation must be something the model cannot forge."""

    def test_it_binds_tool_args_and_turn(self) -> None:
        seen = []
        r = ToolRegistry(confirm=lambda s, a, n: seen.append(n) or True)
        r.register(spec(tier=Tier.YELLOW, model=IndexArgs))
        r.call("ping", {"index": 1}, turn_id=1)
        r.call("ping", {"index": 2}, turn_id=1)
        r.call("ping", {"index": 1}, turn_id=2)
        assert len(set(seen)) == 3, "different args or turns must give different nonces"

    def test_it_is_stable_for_the_same_call(self) -> None:
        r = ToolRegistry(confirm=lambda *_: True)
        s = r.register(spec(tier=Tier.YELLOW, model=IndexArgs))
        _, parsed = r.validate("ping", {"index": 7})
        assert ToolRegistry.nonce(s, parsed, 3) == ToolRegistry.nonce(s, parsed, 3)


class TestBudgets:
    def _clock(self):
        self._t = 0.0

        def now() -> float:
            return self._t

        return now

    def test_a_per_tool_bucket_stops_a_loop(self) -> None:
        r = ToolRegistry(clock=self._clock())
        r.register(spec(max_per_minute=3))
        for _ in range(3):
            r.call("ping", {})
        with pytest.raises(RateLimited):
            r.call("ping", {})

    def test_the_bucket_refills_with_time(self) -> None:
        r = ToolRegistry(clock=self._clock())
        r.register(spec(max_per_minute=2))
        r.call("ping", {})
        r.call("ping", {})
        self._t += 61
        assert r.call("ping", {}) == "ok"

    def test_a_global_budget_covers_many_allowed_tools(self) -> None:
        # Many individually-allowed actions must not add up to a runaway.
        r = ToolRegistry(confirm=lambda *_: True, clock=self._clock())
        for i in range(10):
            r.register(spec(name=f"t{i}", tier=Tier.YELLOW, model=IndexArgs, max_per_minute=100))
        with pytest.raises(RateLimited):
            for i in range(60):
                r.call(f"t{i % 10}", {"index": 0})

    def test_green_reads_do_not_consume_the_side_effect_budget(self) -> None:
        # Asking the battery level 40 times is pointless, not dangerous.
        r = ToolRegistry(clock=self._clock())
        for i in range(10):
            r.register(spec(name=f"g{i}", max_per_minute=100))
        for i in range(100):
            r.call(f"g{i % 10}", {})


class TestInjectionPayloads:
    """The payloads proven during research to execute inside the
    compositor, as negative cases. They cannot even be *expressed* here:
    every tool takes an int or an enum, so there is no field to put them
    in — which is the actual defence, not a filter.
    """

    PAYLOADS = (
        'special:magic"] os.execute("id") --',
        "2; os.execute('id')",
        '1"]=nil; os.execute("touch /tmp/pwned") --',
    )

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_no_registered_tool_accepts_a_lua_payload(self, payload: str) -> None:
        r = ToolRegistry(confirm=lambda *_: True)
        r.register(spec(name="focus", tier=Tier.YELLOW, model=IndexArgs))
        r.register(spec(name="brightness", tier=Tier.YELLOW, model=EnumArgs))
        for tool, key in (("focus", "index"), ("brightness", "direction")):
            with pytest.raises(ToolRejected):
                r.call(tool, {key: payload})

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_a_payload_smuggled_as_an_extra_key_is_rejected(self, payload: str) -> None:
        r = ToolRegistry(confirm=lambda *_: True)
        r.register(spec(name="focus", tier=Tier.YELLOW, model=IndexArgs))
        with pytest.raises(ToolRejected):
            r.call("focus", {"index": 0, "cmd": payload})


class TestAudit:
    """`call()` is the one place every gated action passes through, so
    it is the one place that can promise an `attempted` line before the
    handler runs and an outcome after. A caller that had to remember to
    log would one day forget, and the tool that hung would be the one
    with no record.
    """

    def _log(self, tmp_path) -> AuditLog:
        return AuditLog(path=tmp_path / "audit.jsonl")

    def test_a_green_call_is_bracketed_by_attempt_and_success(self, tmp_path) -> None:
        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(handler=lambda **_: "battery 96 percent"))
        r.call("ping", {}, turn_id=4)
        assert [e["event"] for e in log.entries] == [ATTEMPTED, SUCCEEDED]
        attempt, outcome = log.entries
        assert attempt["turn"] == 4 and attempt["tier"] == "green"
        assert outcome["call"] == attempt["call"]
        assert "96" in outcome["result"]
        assert log.unfinished() == []

    def test_the_attempt_carries_the_validated_arguments(self, tmp_path) -> None:
        # What the log records is what the handler received — after
        # validation filled the defaults — not the raw dict the model sent.
        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(model=EnumArgs))
        r.call("ping", {"direction": "up"})
        assert log.entries[0]["args"] == {"direction": "up", "steps": 1}

    def test_a_denial_is_an_outcome_not_a_gap(self, tmp_path) -> None:
        log = self._log(tmp_path)
        r = ToolRegistry(confirm=lambda *_: False, audit=log)
        r.register(spec(tier=Tier.YELLOW, model=IndexArgs))
        with pytest.raises(ToolNotConfirmed):
            r.call("ping", {"index": 1})
        assert [e["event"] for e in log.entries] == [ATTEMPTED, DENIED]
        assert log.unfinished() == []

    def test_no_confirmer_is_recorded_as_denied(self, tmp_path) -> None:
        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(tier=Tier.YELLOW, model=IndexArgs))
        with pytest.raises(ToolNotConfirmed):
            r.call("ping", {"index": 1})
        assert log.entries[-1]["event"] == DENIED

    def test_a_rate_limit_is_an_outcome(self, tmp_path) -> None:
        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(max_per_minute=1))
        r.call("ping", {})
        with pytest.raises(RateLimited):
            r.call("ping", {})
        assert [e["event"] for e in log.entries] == [ATTEMPTED, SUCCEEDED, ATTEMPTED, RATE_LIMITED]

    def test_a_handler_that_raises_is_recorded_then_re_raised(self, tmp_path) -> None:
        # The log answers "did it run?"; the caller still decides what
        # she says about it, so the exception must come through.
        def boom(**_) -> str:
            raise OSError("playerctl: no players found")

        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(handler=boom))
        with pytest.raises(OSError):
            r.call("ping", {})
        assert [e["event"] for e in log.entries] == [ATTEMPTED, FAILED]
        assert "no players" in log.entries[-1]["error"]

    def test_a_rejected_call_writes_nothing(self, tmp_path) -> None:
        # Nothing ran and nothing could have, and the only arguments to
        # record would be the unvalidated ones — the one way free text
        # could reach a log that promises it holds none.
        log = self._log(tmp_path)
        r = ToolRegistry(audit=log)
        r.register(spec(model=IndexArgs))
        with pytest.raises(ToolRejected):
            r.call("ping", {"index": 'os.execute("id")'})
        with pytest.raises(ToolRejected):
            r.call("nope", {})
        assert log.entries == []

    def test_no_audit_log_is_fine(self) -> None:
        r = ToolRegistry()
        r.register(spec())
        assert r.call("ping", {}) == "ok"


class TestCanConfirm:
    def test_reports_whether_a_yellow_tool_could_ever_run(self) -> None:
        assert ToolRegistry(confirm=lambda *_: True).can_confirm
        assert not ToolRegistry().can_confirm


class TestReachGate:
    """A tool that leaves the machine, on a turn that is already remote.

    The gap this closes was flagged in docs/DECISIONS.md rather than
    silently skipped: `open_app(target="pc")` and `web_search` were
    added on Yash's direct request and consulted only their permission
    tier, never the turn's NETWORK tier — so nothing stopped the model
    reaching the box again, or the internet, through a tunnel.
    """

    def _registry(self, tier: StateTier, reach: Reach, calls: list) -> ToolRegistry:
        reg = ToolRegistry(
            confirm=lambda *a: calls.append("confirmed") or True,
            network_tier=lambda: tier,
        )
        reg.register(
            ToolSpec(
                name="reaching",
                description="d",
                tier=Tier.YELLOW,
                args_model=NoArgs,
                handler=lambda: calls.append("ran") or "done",
                reach=reach,
            )
        )
        return reg

    def test_an_off_machine_tool_runs_at_home(self) -> None:
        for tier in (StateTier.LOCAL, StateTier.LAN):
            calls: list = []
            reg = self._registry(tier, Reach.INTERNET, calls)
            assert reg.call("reaching", {}) == "done"
            assert "ran" in calls, tier

    def test_an_off_machine_tool_is_refused_through_the_tunnel(self) -> None:
        calls: list = []
        reg = self._registry(StateTier.TUNNEL, Reach.INTERNET, calls)
        with pytest.raises(ReachRefused):
            reg.call("reaching", {})
        assert "ran" not in calls, "the handler must not have run"

    def test_a_local_tool_is_unaffected_by_the_tunnel(self) -> None:
        # The regression that would matter most: gating network reach
        # must not quietly disable every ordinary tool away from home.
        calls: list = []
        reg = self._registry(StateTier.TUNNEL, Reach.LOCAL, calls)
        assert reg.call("reaching", {}) == "done"
        assert "ran" in calls

    def test_refusal_does_not_ask_yash_to_confirm(self) -> None:
        # Putting a confirmation in front of him for a call that can
        # never complete trains him to dismiss confirmations.
        calls: list = []
        reg = self._registry(StateTier.TUNNEL, Reach.OWN_MACHINES, calls)
        with pytest.raises(ReachRefused):
            reg.call("reaching", {})
        assert "confirmed" not in calls

    def test_refusal_does_not_spend_the_rate_limit(self) -> None:
        calls: list = []
        reg = self._registry(StateTier.TUNNEL, Reach.INTERNET, calls)
        for _ in range(20):
            with pytest.raises(ReachRefused):
                reg.call("reaching", {})
        # Had refusals consumed bucket slots, this would now be
        # RateLimited instead — a misleading reason for the real cause.
        with pytest.raises(ReachRefused):
            reg.call("reaching", {})

    def test_the_tier_is_read_per_call_not_frozen_at_construction(self) -> None:
        # The turn's tier is snapshotted at turn start, so a registry
        # built once at boot must ask again every call.
        current = StateTier.TUNNEL
        calls: list = []
        reg = ToolRegistry(confirm=lambda *a: True, network_tier=lambda: current)
        reg.register(
            ToolSpec(
                name="reaching",
                description="d",
                tier=Tier.YELLOW,
                args_model=NoArgs,
                handler=lambda: calls.append("ran") or "done",
                reach=Reach.INTERNET,
            )
        )
        with pytest.raises(ReachRefused):
            reg.call("reaching", {})
        current = StateTier.LOCAL
        assert reg.call("reaching", {}) == "done"

    def test_refusal_is_audited_with_the_real_reason(self, tmp_path) -> None:
        calls: list = []
        reg = ToolRegistry(
            confirm=lambda *a: True,
            network_tier=lambda: StateTier.TUNNEL,
            audit=AuditLog(tmp_path / "audit.jsonl"),
        )
        reg.register(
            ToolSpec(
                name="reaching",
                description="d",
                tier=Tier.YELLOW,
                args_model=NoArgs,
                handler=lambda: calls.append("ran") or "done",
                reach=Reach.INTERNET,
            )
        )
        with pytest.raises(ReachRefused):
            reg.call("reaching", {})
        written = (tmp_path / "audit.jsonl").read_text()
        assert DENIED in written
        assert "internet" in written and "tunnel" in written


class TestSourceGate:
    """v2 puts a second caller behind the registry. Until now every call
    came from the LLM, having passed a system prompt, a tool schema and
    (for YELLOW) a confirmation. A gesture has passed none of those, and
    the recogniser cannot tell a deliberate swipe from an identical
    accidental one — so eligibility is opt-in per tool.
    """

    def _registry(self, calls: list, *, gesture_ok: bool) -> ToolRegistry:
        reg = ToolRegistry(confirm=lambda *a: True)
        reg.register(
            ToolSpec(
                name="swipe",
                description="d",
                tier=Tier.GREEN,
                args_model=NoArgs,
                handler=lambda: calls.append("ran") or "done",
                gesture_ok=gesture_ok,
            )
        )
        return reg

    def test_voice_is_the_default_so_existing_callers_are_unchanged(self) -> None:
        calls: list = []
        reg = self._registry(calls, gesture_ok=False)
        assert reg.call("swipe", {}) == "done"
        assert calls == ["ran"]

    def test_a_gesture_cannot_reach_a_tool_that_did_not_opt_in(self) -> None:
        calls: list = []
        reg = self._registry(calls, gesture_ok=False)
        with pytest.raises(SourceRefused):
            reg.call("swipe", {}, source=Source.GESTURE)
        assert calls == [], "the handler must not have run"

    def test_a_gesture_reaches_a_tool_that_opted_in(self) -> None:
        calls: list = []
        reg = self._registry(calls, gesture_ok=True)
        assert reg.call("swipe", {}, source=Source.GESTURE) == "done"
        assert calls == ["ran"]

    def test_refusal_is_audited_with_the_source(self, tmp_path) -> None:
        reg = ToolRegistry(confirm=lambda *a: True, audit=AuditLog(tmp_path / "a.jsonl"))
        reg.register(
            ToolSpec(
                name="swipe",
                description="d",
                tier=Tier.GREEN,
                args_model=NoArgs,
                handler=lambda: "done",
            )
        )
        with pytest.raises(SourceRefused):
            reg.call("swipe", {}, source=Source.GESTURE)
        written = (tmp_path / "a.jsonl").read_text()
        assert DENIED in written and "gesture" in written

    def test_gesture_spam_does_not_exhaust_the_voice_budget(self) -> None:
        # The bug this prevents: buckets keyed on tool name alone mean a
        # camera misreading a wave burns the LLM's allowance for the same
        # tool, and the voice path is rate-limited for something it did
        # not do.
        calls: list = []
        reg = self._registry(calls, gesture_ok=True)
        for _ in range(6):
            reg.call("swipe", {}, source=Source.GESTURE)
        with pytest.raises(RateLimited):
            reg.call("swipe", {}, source=Source.GESTURE)
        # The voice path still has its own full allowance.
        assert reg.call("swipe", {}) == "done"

    def test_the_gesture_bucket_is_still_a_ceiling(self) -> None:
        # Separating the buckets must not mean the camera is unlimited —
        # a misread wave becoming forty swipes is exactly the runaway
        # these buckets exist to stop.
        calls: list = []
        reg = self._registry(calls, gesture_ok=True)
        for _ in range(6):
            reg.call("swipe", {}, source=Source.GESTURE)
        with pytest.raises(RateLimited):
            reg.call("swipe", {}, source=Source.GESTURE)
