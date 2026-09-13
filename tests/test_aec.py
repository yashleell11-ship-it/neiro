"""Echo cancellation: the preconditions, not the algorithm.

This module deliberately implements no cancellation — PipeWire's WebRTC
AEC3 is already on disk and a hand-rolled Python one would be strictly
worse. What it does is check the things that make a canceller look
broken when they are actually the problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from neiro.audio.aec import CLIPPED_FRACTION, DROP_IN, check_gain, print_config, suppression_db


class TestGainIsCheckedFirst:
    def test_a_clipped_input_is_flagged(self) -> None:
        # Measured on this laptop 2026-09-11: Capture at +30 dB with mic
        # boost on gave RMS 0.9-0.95 for an entire 3-second clip. A
        # canceller subtracts a reference from the mic signal and cannot
        # recover a waveform whose information is already gone.
        saturated = np.ones(16000, dtype=np.float32)
        check = check_gain(saturated)
        assert check.saturated
        assert "amixer" in (check.remedy() or "")

    def test_a_healthy_input_is_not_flagged(self) -> None:
        t = np.linspace(0, 1, 16000, endpoint=False)
        healthy = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        check = check_gain(healthy)
        assert not check.saturated
        assert check.remedy() is None

    def test_occasional_peaks_are_tolerated(self) -> None:
        # Real speech touches full scale sometimes. The bar is sustained
        # saturation, not a single loud syllable.
        audio = np.zeros(16000, dtype=np.float32)
        audio[:50] = 1.0
        assert not check_gain(audio).saturated
        assert CLIPPED_FRACTION > 0

    def test_empty_input_does_not_crash(self) -> None:
        assert check_gain(np.zeros(0, dtype=np.float32)).peak == 0.0


class TestSuppression:
    def test_a_quieter_cancelled_path_scores_positive_db(self) -> None:
        raw = np.ones(1000, dtype=np.float32)
        cancelled = np.full(1000, 0.1, dtype=np.float32)
        assert suppression_db(raw, cancelled) == pytest.approx(20.0)

    def test_no_suppression_is_zero_db(self) -> None:
        raw = np.full(1000, 0.5, dtype=np.float32)
        assert suppression_db(raw, raw) == 0.0

    def test_silence_out_is_infinite_suppression(self) -> None:
        raw = np.ones(1000, dtype=np.float32)
        assert suppression_db(raw, np.zeros(1000, dtype=np.float32)) == float("inf")


class TestConfig:
    def test_only_valid_webrtc_keys_are_used(self) -> None:
        # Anything else is silently ignored by PipeWire, which looks
        # exactly like a canceller that does not work.
        valid = {
            "webrtc.gain_control",
            "webrtc.noise_suppression",
            "webrtc.high_pass_filter",
            "webrtc.echo_canceller",
            "webrtc.voice_detection",
            "webrtc.extended_filter",
        }
        used = {line.strip().split(" =")[0] for line in DROP_IN.splitlines() if "webrtc." in line}
        assert used <= valid

    def test_gain_control_is_off(self) -> None:
        # It fights the level the canceller is trying to match.
        assert "webrtc.gain_control = false" in DROP_IN

    def test_suspend_timeout_is_disabled_on_neiros_nodes(self) -> None:
        # PipeWire suspends idle nodes after 5 s, and a canceller that
        # has just resumed has not adapted -- so the first thing she says
        # after a pause echoes.
        assert "session.suspend-timeout-seconds = 0" in DROP_IN

    def test_the_config_is_printed_never_written(self) -> None:
        # It changes how every application on the machine records audio,
        # which is not a voice assistant's decision to make alone.
        text = print_config()
        assert "Save as" in text and "systemctl --user restart pipewire" in text


class TestRawAudioForAffect:
    def test_the_module_says_affect_reads_the_raw_mic(self) -> None:
        # AEC's noise suppression destroys the prosody features Lane A
        # measures. Getting this backwards would make her emotionally
        # blind exactly when speakers are in use.
        from neiro.audio import aec

        assert "RAW" in aec.__doc__
        assert "prosody" in aec.__doc__
