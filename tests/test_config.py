"""Regression test for the TOML config source.

Real bug hit while running Stage 0 Task 0 for the first time (2026-09-11):
`Neiro.model_config` declared `toml_file=...` but the class never
overrode `settings_customise_sources`, so pydantic-settings silently
ignored it — with only a one-line UserWarning as the tell. `neiro doctor`
would report "audio device profiles named: FAIL" forever, even with a
correctly written config.toml on disk, because it was never being read.

This test writes a temp config.toml, points a Neiro subclass at it, and
asserts the values actually load — so if the TOML source is ever
unwired again, this fails loudly instead of degrading to a silent
UserWarning nobody reads.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from neiro.config import Neiro


def test_toml_file_is_actually_read(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        schema_version = 1

        [audio]
        active_profile = "earbuds"

        [audio.profiles.earbuds]
        input_device = "bluez_input.aa:bb:cc"
        output_device = "bluez_output.aa_bb_cc.1"
        """
    )

    class _NeiroForTest(Neiro):
        model_config = SettingsConfigDict(toml_file=str(config_path))

    cfg = _NeiroForTest()

    assert cfg.audio.active_profile == "earbuds"
    assert "earbuds" in cfg.audio.profiles
    assert cfg.audio.profiles["earbuds"].input_device == "bluez_input.aa:bb:cc"


def test_active_profile_must_live_under_the_audio_table(tmp_path: Path) -> None:
    """The TOML footgun `neiro doctor --audio-inventory` now warns about:
    a bare `active_profile = ...` placed AFTER a nested table header
    attaches to that table, not to [audio]. This documents the failure
    mode so nobody "fixes" the warning text without understanding why
    it's there.
    """

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        schema_version = 1

        [audio.profiles.earbuds]
        input_device = "bluez_input.aa:bb:cc"
        output_device = "bluez_output.aa_bb_cc.1"

        active_profile = "earbuds"
        """
    )

    class _NeiroForTest(Neiro):
        model_config = SettingsConfigDict(toml_file=str(config_path))

    cfg = _NeiroForTest()

    # This is the bug, captured as a test: active_profile did NOT land
    # where a naive reading of the file would suggest.
    assert cfg.audio.active_profile == ""
