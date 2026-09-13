"""Tests for the Hyprland tool layer.

Focused on the safety property rather than the socket: whatever the
model supplies, the payload that reaches the compositor is built from
Neiro's own resolved values and survives the egress filter. The socket
is replaced by a capture here — nothing in the repo drives it live yet,
so the wire string itself is what these pin, because that is the part
every online example gets wrong (CLAUDE.md rule 1).
"""

from __future__ import annotations

import pytest

from neiro.tools import hyprland
from neiro.tools.egress import EgressRejected, check
from neiro.tools.hyprland import Window, dispatch, focus_window_by_index, switch_workspace


def fake_windows() -> list[Window]:
    return [
        Window(index=0, address="0xaaa111", app="kitty", title="~/code", workspace=1),
        Window(index=1, address="0xbbb222", app="zen", title="GitHub", workspace=2),
    ]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Everything that would have gone down the socket, in order.

    `dispatch()` looks `request` up on the module at call time, so
    replacing it here is enough — and the fake never opens a socket,
    which matters because a real one would freeze the desktop if left
    open (module docstring of hyprland.py).
    """
    sent: list[str] = []

    def fake_request(command: str, timeout_s: float = hyprland.SOCKET_TIMEOUT_S) -> str:
        sent.append(command)
        return "ok"

    monkeypatch.setattr(hyprland, "request", fake_request)
    return sent


class TestWindow:
    def test_describe_includes_app_and_title(self) -> None:
        w = fake_windows()[0]
        assert "kitty" in w.describe()
        assert "~/code" in w.describe()

    def test_describe_handles_missing_title(self) -> None:
        w = Window(index=0, address="0x1", app="steam", title="", workspace=1)
        assert w.describe() == "steam"


class TestIndexResolution:
    """The model only ever supplies an integer. These confirm that a bad
    integer is a plain Python error on our side, never a command with a
    hostile string in it.
    """

    def test_out_of_range_index_raises_cleanly(self) -> None:
        with pytest.raises(IndexError):
            focus_window_by_index(99, fake_windows())

    def test_negative_index_raises_rather_than_wrapping(self) -> None:
        # Python would happily treat -1 as "last item"; for a voice
        # command that's a silent wrong-window, so it's rejected.
        with pytest.raises(IndexError):
            focus_window_by_index(-1, fake_windows())

    def test_empty_window_list(self) -> None:
        with pytest.raises(IndexError):
            focus_window_by_index(0, [])

    def test_suspicious_address_is_refused(self) -> None:
        # Defence in depth: if the window table itself were ever poisoned
        # (a compromised compositor response), refuse rather than
        # interpolate it.
        poisoned = [Window(index=0, address='"}); os.execute("x"', app="x", title="", workspace=1)]
        with pytest.raises((ValueError, EgressRejected)):
            focus_window_by_index(0, poisoned)


class TestWorkspaceValidation:
    @pytest.mark.parametrize("number", [0, -1, 11, 999])
    def test_out_of_range_workspace_refused(self, number: int) -> None:
        with pytest.raises(ValueError):
            switch_workspace(number)


class TestGeneratedPayloadsSurviveEgress:
    """Whatever these tools build must pass the filter they're behind —
    otherwise the filter would be rejecting legitimate traffic and
    someone would eventually be tempted to loosen it.

    These call the real generators and read the wire string back rather
    than asserting against hand-written literals. A literal that
    duplicates the implementation cannot notice when the implementation
    drifts — and the drift this guards against is specific: rewriting
    either generator to the hyprlang form (`dispatch workspace 2`) that
    every LLM completion emits left the whole suite green before this.
    """

    def test_focus_payload_is_lua_and_passes_the_filter(self, wire: list[str]) -> None:
        focus_window_by_index(0, fake_windows())
        assert wire == ['dispatch hl.dsp.focus({window="address:0xaaa111"})']
        payload = wire[0].removeprefix("dispatch ")
        assert check(payload) == payload

    def test_index_resolves_to_that_window_not_the_first(self, wire: list[str]) -> None:
        focus_window_by_index(1, fake_windows())
        assert wire == ['dispatch hl.dsp.focus({window="address:0xbbb222"})']

    @pytest.mark.parametrize("n", range(1, 11))
    def test_workspace_payload_is_lua_and_passes_the_filter(self, wire: list[str], n: int) -> None:
        switch_workspace(n)
        assert wire == [f"dispatch hl.dsp.focus({{workspace={n}}})"]
        payload = wire[0].removeprefix("dispatch ")
        assert check(payload) == payload


class TestDispatchRunsTheEgressFilter:
    """Wall 3 is only a wall if it is in the write path.

    Deleting the `egress_check(payload)` line in dispatch() used to leave
    every test green (ruff's unused-import warning was the only thing
    noticing). Each case here records what was sent, so a filter that
    is bypassed shows up as a send that must not have happened.
    """

    def test_denylisted_payload_never_reaches_the_socket(self, wire: list[str]) -> None:
        with pytest.raises(EgressRejected, match="exec"):
            dispatch('hl.dsp.exec({cmd="poweroff"})')
        assert wire == []

    def test_well_formed_red_dispatcher_never_reaches_the_socket(self, wire: list[str]) -> None:
        # Shape-valid, so only the dispatcher allowlist stands between
        # this and the compositor.
        with pytest.raises(EgressRejected, match="dispatcher"):
            dispatch("hl.dsp.killactive({})")
        assert wire == []

    def test_dispatch_obeys_the_filter_not_its_own_opinion(
        self, wire: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A payload the tools legitimately generate is still refused when
        # the filter says no: dispatch() defers to it unconditionally.
        def veto(payload: str) -> str:
            raise EgressRejected(f"vetoed by test: {payload}")

        monkeypatch.setattr(hyprland, "egress_check", veto)
        with pytest.raises(EgressRejected, match="vetoed by test"):
            switch_workspace(3)
        assert wire == []
