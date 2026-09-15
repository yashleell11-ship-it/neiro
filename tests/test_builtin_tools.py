"""The real tools, as registered.

These run against this actual machine where they can, because a tool
that type-checks and then returns nothing useful is the failure this
project keeps finding (see `adjust_brightness`, which was written
against a script interface that did not exist).
"""

from __future__ import annotations

from typing import Literal, get_args, get_origin

import pytest

from elizabeth.tools import apps, media, system, websearch
from elizabeth.tools.builtin import build_registry
from elizabeth.tools.registry import ToolRegistry, ToolRejected
from elizabeth.tools.tiers import Tier

GREEN_TOOLS = ("system_stats", "audio_state", "now_playing", "list_windows", "active_window")
YELLOW_TOOLS = (
    "set_volume",
    "set_mute",
    "adjust_brightness",
    "media_control",
    "focus_window",
    "switch_workspace",
    "open_app",
    "web_search",
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
        # The one legal exception is a field opted into
        # spoken_text_fields (registry.py) — genuinely spoken content,
        # never an identifier. web_search.query is the first of these.
        for name in registry.names():
            spec = registry.spec(name)
            for field, info in spec.args_model.model_fields.items():
                if field in spec.spoken_text_fields:
                    assert info.annotation is str, f"{name}.{field} is spoken but not str"
                    continue
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


class TestAuditPassthrough:
    def test_the_log_handed_in_is_the_one_written(self, tmp_path) -> None:
        # The daemon hands the registry its audit log; a registry that
        # quietly made its own would write to ~/.local/state from a test
        # and record nothing where the daemon looks.
        from elizabeth.tools.audit import ATTEMPTED, SUCCEEDED, AuditLog

        log = AuditLog(path=tmp_path / "audit.jsonl")
        build_registry(confirm=None, audit=log).call("audio_state", {}, turn_id=2)
        assert [e["event"] for e in log.entries] == [ATTEMPTED, SUCCEEDED]
        assert log.entries[0]["tool"] == "audio_state" and log.entries[0]["turn"] == 2
        assert (tmp_path / "audit.jsonl").exists()


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
        from elizabeth.tools.registry import ToolNotConfirmed

        # Perfectly well-formed and still must not run unasked.
        with pytest.raises(ToolNotConfirmed):
            registry.call("set_volume", {"percent": 30})


def _advertised_enum(registry: ToolRegistry, tool: str, field: str) -> list[str]:
    """The enum exactly as the model sees it, read back from the schema
    rather than from the pydantic class — the schema is the contract.
    """
    schema = next(s for s in registry.schemas() if s["function"]["name"] == tool)
    return schema["function"]["parameters"]["properties"][field]["enum"]


def _description(registry: ToolRegistry, tool: str) -> str:
    schema = next(s for s in registry.schemas() if s["function"]["name"] == tool)
    return schema["function"]["description"]


def _confirming() -> ToolRegistry:
    """A fresh, auto-confirming registry.

    Fresh per call on purpose: `ToolSpec.max_per_minute` is a real limit
    on a real registry, and an enum with more members than the bucket
    allows would otherwise read as RateLimited rather than as the drift
    these tests exist to catch.
    """
    return build_registry(confirm=lambda *_: True)


class TestAdvertisedEnumsAreExecutable:
    """The schema is prompt text: whatever it advertises, the model will
    eventually emit. An enum member the handler then refuses is the worst
    failure a tool can have — it passes validation, spends the rate-limit
    bucket and the side-effect budget, gets Yash to click Yes, and only
    then dies, with a ValueError the registry's ToolError contract does
    not cover. That happened with `media_control("stop")`: promised in
    the description and the enum, absent from `media.ACTIONS`.
    """

    @pytest.fixture
    def playerctl_argv(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[list[str]]:
        # Stub the one subprocess seam in each module, so the suite never
        # actually pauses what is playing or mutes the speakers. The list
        # records what media.py would have run.
        argv: list[list[str]] = []

        def fake_playerctl(args: list[str], timeout: float = 0.0) -> tuple[str, bool]:
            argv.append(list(args))
            return "", True

        monkeypatch.setattr(media, "_run", fake_playerctl)
        monkeypatch.setattr(system, "_run", lambda cmd, timeout=0.0: "")
        # adjust_brightness refuses to run without its script; the test is
        # about the enum, not about whether this machine has the script.
        script = tmp_path / "brightness-smart.sh"
        script.touch()
        monkeypatch.setattr(system, "BRIGHTNESS_SCRIPT", script)
        # open_app would really launch a process (locally) or ssh to the
        # box; web_search would really hit the network. Neither belongs
        # in a unit suite — stub both to the boundary this module owns.
        monkeypatch.setattr(apps, "open_app", lambda target, app: f"stub open {app} {target}")
        monkeypatch.setattr(websearch, "search_and_describe", lambda query: f"stub search {query}")
        return argv

    def test_media_schema_and_allowlist_are_one_set(self, registry: ToolRegistry) -> None:
        # Both directions: nothing advertised that cannot run, nothing
        # runnable that the model is not told about.
        assert set(_advertised_enum(registry, "media_control", "action")) == set(media.ACTIONS)

    def test_media_description_promises_nothing_outside_the_enum(
        self, registry: ToolRegistry
    ) -> None:
        # The description is the model's only basis for choosing the
        # tool. It said "stop" while the enum could not deliver it.
        text = _description(registry, "media_control").lower()
        enum = set(_advertised_enum(registry, "media_control", "action"))
        for verb in ("stop", "open", "position"):
            assert verb in enum or verb not in text, verb

    def test_every_advertised_media_action_reaches_playerctl(
        self, playerctl_argv: list[list[str]]
    ) -> None:
        for action in _advertised_enum(_confirming(), "media_control", "action"):
            playerctl_argv.clear()
            result = _confirming().call("media_control", {"action": action})
            assert isinstance(result, str) and result.strip(), action
            # The enum member is the playerctl verb, passed through
            # unmodified — no mapping in between that could drift.
            assert playerctl_argv == [[action]], action

    def test_every_literal_member_of_every_tool_is_accepted(
        self, playerctl_argv: list[list[str]]
    ) -> None:
        # The class of bug, not the instance: for every tool, every
        # member of every Literal field must run without a bare
        # ValueError. A tool that needs another required argument will
        # be rejected here, loudly — extend the test, do not skip it.
        shape = _confirming()
        checked = 0
        for name in shape.names():
            spec = shape.spec(name)
            literal_fields = {
                field: info
                for field, info in spec.args_model.model_fields.items()
                if get_origin(info.annotation) is Literal
            }
            for field, info in literal_fields.items():
                # A tool can have more than one required Literal field
                # (open_app: target AND app) — calling with only the one
                # under test would fail on the other's missing value, not
                # on the thing this test is actually checking. Every
                # OTHER required field gets its first enum member so the
                # call is well-formed; the field under test still varies
                # across its full range.
                other_defaults = {
                    other_field: get_args(other_info.annotation)[0]
                    for other_field, other_info in literal_fields.items()
                    if other_field != field
                }
                for member in get_args(info.annotation):
                    args = {**other_defaults, field: member}
                    result = _confirming().call(name, args)
                    assert isinstance(result, str) and result.strip(), f"{name}.{field}={member!r}"
                    checked += 1
        # If this ever reads zero the test has stopped testing anything.
        assert checked >= len(media.ACTIONS)

    @pytest.mark.parametrize("action", ["stop", "open", "position", "stop ", "", "play; ls"])
    def test_a_bad_action_is_a_tool_error_and_never_reaches_playerctl(
        self, playerctl_argv: list[list[str]], action: str
    ) -> None:
        # ToolRejected, not ValueError: the registry's callers speak
        # ToolError messages aloud and are not written for anything else.
        # "stop" is the case that used to get all the way to the handler.
        with pytest.raises(ToolRejected):
            _confirming().call("media_control", {"action": action})
        assert playerctl_argv == []
