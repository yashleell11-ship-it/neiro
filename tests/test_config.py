"""Regression test for the TOML config source.

Real bug hit while running Stage 0 Task 0 for the first time (2026-09-11):
`Elizabeth.model_config` declared `toml_file=...` but the class never
overrode `settings_customise_sources`, so pydantic-settings silently
ignored it — with only a one-line UserWarning as the tell. `elizabeth doctor`
would report "audio device profiles named: FAIL" forever, even with a
correctly written config.toml on disk, because it was never being read.

This test writes a temp config.toml, points an Elizabeth subclass at it, and
asserts the values actually load — so if the TOML source is ever
unwired again, this fails loudly instead of degrading to a silent
UserWarning nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from elizabeth.config import Elizabeth


def _elizabeth_reading(config_path: Path) -> Elizabeth:
    """An Elizabeth that reads THIS file, not ~/.config/elizabeth/config.toml —
    so the test says nothing about what is on this machine.
    """

    class _ElizabethForTest(Elizabeth):
        model_config = SettingsConfigDict(toml_file=str(config_path))

    return _ElizabethForTest()


def test_stt_language_defaults_to_auto(tmp_path: Path) -> None:
    # Hindi and English are equal priorities and he switches
    # mid-sentence; the default must not pin either.
    config_path = tmp_path / "config.toml"
    config_path.write_text("schema_version = 1\n")
    assert _elizabeth_reading(config_path).stt.language == "auto"


def test_stt_language_is_read_from_the_stt_table(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('schema_version = 1\n\n[stt]\nlanguage = "hi"\n')
    assert _elizabeth_reading(config_path).stt.language == "hi"


def test_stt_language_rejects_a_code_neither_recogniser_serves(tmp_path: Path) -> None:
    # A typo here would otherwise reach faster-whisper as a language
    # code it may or may not know, and fail on the first utterance
    # instead of at load.
    config_path = tmp_path / "config.toml"
    config_path.write_text('schema_version = 1\n\n[stt]\nlanguage = "fr"\n')
    with pytest.raises(ValidationError):
        _elizabeth_reading(config_path)


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

    class _ElizabethForTest(Elizabeth):
        model_config = SettingsConfigDict(toml_file=str(config_path))

    cfg = _ElizabethForTest()

    assert cfg.audio.active_profile == "earbuds"
    assert "earbuds" in cfg.audio.profiles
    assert cfg.audio.profiles["earbuds"].input_device == "bluez_input.aa:bb:cc"


def test_active_profile_must_live_under_the_audio_table(tmp_path: Path) -> None:
    """The TOML footgun `elizabeth doctor --audio-inventory` now warns about:
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

    class _ElizabethForTest(Elizabeth):
        model_config = SettingsConfigDict(toml_file=str(config_path))

    cfg = _ElizabethForTest()

    # This is the bug, captured as a test: active_profile did NOT land
    # where a naive reading of the file would suggest.
    assert cfg.audio.active_profile == ""
