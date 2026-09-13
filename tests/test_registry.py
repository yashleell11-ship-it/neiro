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

from neiro.tools.registry import (
    RateLimited,
    ToolNotConfirmed,
    ToolRegistry,
    ToolRejected,
    ToolSpec,
)
from neiro.tools.tiers import Tier


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
