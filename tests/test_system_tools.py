"""Tests for the Green/Yellow system and media tools.

The validation tests matter most: every Yellow action takes an int from
a constrained range or a Literal enum member, so there is never a string
from the model that reaches a command line. These assert that the
validation actually rejects, rather than trusting the type hint.
"""

from __future__ import annotations

import pytest

from elizabeth.tools.media import ACTIONS, NowPlaying, get_now_playing, media_control
from elizabeth.tools.system import (
    MAX_BRIGHTNESS_STEPS,
    AudioState,
    SystemStats,
    adjust_brightness,
    get_audio_state,
    get_system_stats,
    set_mute,
    set_volume,
)


class TestSystemStatsAreReal:
    """These run against the actual machine — Green tier is read-only,
    so there's nothing to mock and nothing to break.
    """

    def test_reads_something_real(self) -> None:
        stats = get_system_stats()
        assert isinstance(stats, SystemStats)
        # This laptop has a battery; if this ever fails on other hardware
        # the tool should degrade to None rather than crash, which the
        # next test covers.
        assert stats.memory_percent is not None
        assert 0 <= stats.memory_percent <= 100

    def test_disk_free_is_plausible(self) -> None:
        stats = get_system_stats()
        assert stats.disk_free_gb is None or stats.disk_free_gb >= 0

    def test_missing_hardware_degrades_to_none_not_a_crash(self) -> None:
        stats = get_system_stats(disk_path="/nonexistent/path/xyz")
        assert stats.disk_free_gb is None  # degraded, didn't raise

    def test_describe_is_a_short_spoken_sentence(self) -> None:
        spoken = get_system_stats().describe()
        assert spoken
        assert "[" not in spoken  # no markup leaking into speech
        assert len(spoken.split()) < 25


class TestSystemStatsDescribe:
    def test_mentions_battery_and_charge_state(self) -> None:
        stats = SystemStats(
            battery_percent=94,
            battery_charging=True,
            cpu_percent=10.0,
            memory_percent=40.0,
            disk_free_gb=245.0,
        )
        spoken = stats.describe()
        assert "94" in spoken
        assert "charging" in spoken

    def test_says_on_battery_when_discharging(self) -> None:
        stats = SystemStats(
            battery_percent=50,
            battery_charging=False,
            cpu_percent=None,
            memory_percent=None,
            disk_free_gb=None,
        )
        assert "on battery" in stats.describe()

    def test_degrades_gracefully_when_nothing_is_readable(self) -> None:
        stats = SystemStats(None, None, None, None, None)
        assert "can't read" in stats.describe().lower()


class TestVolumeValidation:
    @pytest.mark.parametrize("percent", [-1, 101, 1000, -100])
    def test_out_of_range_is_refused(self, percent: int) -> None:
        # A misheard "volume one thousand" must not reach wpctl.
        with pytest.raises(ValueError):
            set_volume(percent)

    def test_boundaries_are_accepted(self) -> None:
        # Not asserting the sound actually changed — just that the
        # validation doesn't reject legitimate values.
        assert "0" in set_volume(0)
        assert "100" in set_volume(100)
        set_volume(30)  # leave it somewhere sane


class TestMuteValidation:
    @pytest.mark.parametrize("state", ["yes", "ON", "", "; rm -rf /", "1"])
    def test_only_the_enum_members_are_accepted(self, state: str) -> None:
        with pytest.raises(ValueError):
            set_mute(state)

    def test_valid_states(self) -> None:
        for state in ("on", "off", "toggle"):
            assert state in set_mute(state)
        set_mute("off")  # leave it unmuted


class TestBrightnessValidation:
    @pytest.mark.parametrize("direction", ["brighter", "UP", "", "-i", "; ls"])
    def test_direction_must_be_an_enum_member(self, direction: str) -> None:
        with pytest.raises(ValueError):
            adjust_brightness(direction)

    @pytest.mark.parametrize("steps", [0, -1, MAX_BRIGHTNESS_STEPS + 1, 999])
    def test_steps_are_bounded(self, steps: int) -> None:
        # Bounded so a misheard number can't walk the panel to its floor
        # in one command.
        with pytest.raises(ValueError):
            adjust_brightness("down", steps)


class TestAudioState:
    def test_reads_the_real_sink(self) -> None:
        state = get_audio_state()
        assert isinstance(state, AudioState)
        assert state.volume_percent is None or 0 <= state.volume_percent <= 100

    def test_describe_mentions_muted(self) -> None:
        assert "muted" in AudioState(volume_percent=40, muted=True).describe()

    def test_describe_without_volume(self) -> None:
        assert "can't read" in AudioState(volume_percent=None, muted=None).describe().lower()


class TestMediaValidation:
    @pytest.mark.parametrize("action", ["stop", "open", "position", "", "play; ls"])
    def test_only_allowed_actions(self, action: str) -> None:
        with pytest.raises(ValueError):
            media_control(action)

    def test_allowed_actions_do_not_raise(self) -> None:
        # Nothing is playing on this machine, so these return
        # "nothing's playing" rather than failing — which is the correct
        # behaviour, not an error.
        for action in ACTIONS:
            assert media_control(action)

    def test_open_is_deliberately_not_available(self) -> None:
        # It takes a URI — a string that would reach a command line.
        assert "open" not in ACTIONS


class TestNowPlaying:
    def test_none_when_nothing_is_playing(self) -> None:
        # Normal state on this machine right now, and not a failure.
        result = get_now_playing()
        assert result is None or isinstance(result, NowPlaying)

    def test_describe_artist_and_title(self) -> None:
        np = NowPlaying(artist="Boards of Canada", title="Roygbiv", status="Playing")
        assert np.describe() == "Roygbiv by Boards of Canada"

    def test_describe_mentions_paused(self) -> None:
        np = NowPlaying(artist="X", title="Y", status="Paused")
        assert "paused" in np.describe()

    def test_describe_with_only_a_title(self) -> None:
        np = NowPlaying(artist="", title="some podcast", status="Playing")
        assert np.describe() == "some podcast"
