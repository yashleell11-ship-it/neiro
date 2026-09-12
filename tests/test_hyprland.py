"""Tests for the Hyprland tool layer.

Focused on the safety property rather than the socket: whatever the
model supplies, the payload that reaches the compositor is built from
Neiro's own resolved values and survives the egress filter. The socket
itself is exercised live in scripts/ and by `neiro doctor`.
"""

from __future__ import annotations

import pytest

from neiro.tools.egress import EgressRejected, check
from neiro.tools.hyprland import Window, focus_window_by_index, switch_workspace


def fake_windows() -> list[Window]:
    return [
        Window(index=0, address="0xaaa111", app="kitty", title="~/code", workspace=1),
        Window(index=1, address="0xbbb222", app="zen", title="GitHub", workspace=2),
    ]


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
    """

    def test_focus_payload_shape_is_allowed(self) -> None:
        assert check('hl.dsp.focus({window="address:0xaaa111"})')

    def test_workspace_payload_shape_is_allowed(self) -> None:
        for n in range(1, 11):
            assert check(f"hl.dsp.focus({{workspace={n}}})")
