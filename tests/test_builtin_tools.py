"""The real tools, as registered.

These run against this actual machine where they can, because a tool
that type-checks and then returns nothing useful is the failure this
project keeps finding (see `adjust_brightness`, which was written
against a script interface that did not exist).
"""

from __future__ import annotations

import pytest

from neiro.tools.builtin import build_registry
from neiro.tools.registry import ToolRegistry, ToolRejected
from neiro.tools.tiers import Tier

GREEN_TOOLS = ("system_stats", "audio_state", "now_playing", "list_windows", "active_window")
YELLOW_TOOLS = (
    "set_volume",
    "set_mute",
    "adjust_brightness",
    "media_control",
    "focus_window",
    "switch_workspace",
)


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    # Never auto-confirming in tests: a YELLOW tool that runs here would
    # change the machine's volume or focus mid-suite.
    return build_registry(confirm=None)


class TestShape:
    def test_every_expected_tool_is_registered(self, registry: ToolRegistry) -> None:
        assert set(registry.names()) == set(GREEN_TOOLS) | set(YELLOW_TOOLS)

    def test_reads_are_green_and_changes_are_yellow(self, registry: ToolRegistry) -> None:
        green = {s["function"]["name"] for s in registry.schemas(Tier.GREEN)}
        assert green == set(GREEN_TOOLS)

    def test_no_tool_takes_a_free_form_string(self, registry: ToolRegistry) -> None:
        # The registry enforces this at registration, so building it at
        # all is the assertion — but state it, because it IS the design.
        for name in registry.names():
            spec = registry.spec(name)
            for field, info in spec.args_model.model_fields.items():
                assert info.annotation is not str, f"{name}.{field}"

    def test_descriptions_tell_the_model_when_to_use_them(self, registry: ToolRegistry) -> None:
        # This string is the model's only basis for deciding to call it.
        for schema in registry.schemas():
            assert len(schema["function"]["description"]) > 20

    def test_focus_window_is_documented_as_needing_the_list_first(
        self, registry: ToolRegistry
    ) -> None:
        # The index is only meaningful against a list from this turn —
        # that is the whole defence against the Lua injection.
        text = next(
            s["function"]["description"]
            for s in registry.schemas()
            if s["function"]["name"] == "focus_window"
        )
        assert "list_windows" in text


class TestGreenToolsAgainstThisMachine:
    @pytest.mark.parametrize("name", GREEN_TOOLS)
    def test_it_returns_something_a_person_could_hear(
        self, registry: ToolRegistry, name: str
    ) -> None:
        result = registry.call(name, {})
        assert isinstance(result, str) and result.strip()
        # Spoken output: no markup, no data structures leaking through.
        for forbidden in ("{", "[]", "None", "Traceback", "<"):
            assert forbidden not in result, f"{name}: {result!r}"

    def test_system_stats_reports_real_numbers(self, registry: ToolRegistry) -> None:
        text = registry.call("system_stats", {})
        assert "percent" in text

    def test_brightness_is_relative_because_absolute_does_not_work_here(
        self, registry: ToolRegistry
    ) -> None:
        # The sysfs backlight accepts writes and moves nothing with the
        # GPU MUX in discrete mode; brightness-smart.sh takes only -i/-d.
        schema = next(s for s in registry.schemas() if s["function"]["name"] == "adjust_brightness")
        props = schema["function"]["parameters"]["properties"]
        assert props["direction"]["enum"] == ["up", "down"]
        assert "percent" not in props


class TestRejections:
    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("set_volume", {"percent": 9999}),
            ("set_volume", {"percent": -5}),
            ("set_volume", {"percent": 30, "shell": "rm -rf /"}),
            ("adjust_brightness", {"direction": "sideways"}),
            ("adjust_brightness", {"direction": "down", "steps": 99}),
            ("switch_workspace", {"number": 0}),
            ("switch_workspace", {"number": 99}),
            ("media_control", {"action": "delete"}),
            ("focus_window", {"index": -1}),
        ],
    )
    def test_out_of_range_and_unknown_values_are_refused(
        self, registry: ToolRegistry, name: str, args: dict
    ) -> None:
        with pytest.raises(ToolRejected):
            registry.call(name, args)

    @pytest.mark.parametrize(
        "payload",
        [
            'special:magic"] os.execute("id") --',
            "2; os.execute('id')",
            '1"]=nil; os.execute("touch /tmp/pwned") --',
        ],
    )
    def test_the_proven_lua_payloads_cannot_reach_any_tool(
        self, registry: ToolRegistry, payload: str
    ) -> None:
        # Proven during the research pass to execute inside the
        # compositor. There is no string-typed field to carry them.
        for name, key in (
            ("focus_window", "index"),
            ("switch_workspace", "number"),
            ("set_volume", "percent"),
            ("adjust_brightness", "direction"),
            ("media_control", "action"),
        ):
            with pytest.raises(ToolRejected):
                registry.call(name, {key: payload})

    def test_a_valid_yellow_call_still_needs_confirmation(self, registry: ToolRegistry) -> None:
        from neiro.tools.registry import ToolNotConfirmed

        # Perfectly well-formed and still must not run unasked.
        with pytest.raises(ToolNotConfirmed):
            registry.call("set_volume", {"percent": 30})
