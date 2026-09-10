"""Microphone capture. Stage 0: a blocking record() for `neiro mic-test`
and (later, same stage) `neiro record-set`. Stage 1 adds a streaming
callback API for the always-on VAD loop — deliberately not built yet;
Task 3's whole point is proving the plumbing works before anything
depends on it.

Always 16 kHz mono float32, blocksize 512 — chosen so that later, when
Silero VAD is wired in (Stage 1), one audio callback IS exactly one VAD
window (512 samples / 16000 Hz = 32 ms), removing a whole layer of
buffering code that would otherwise have to chop an arbitrary blocksize
into VAD-sized pieces.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from neiro.audio.devices import apply_to_environment, resolve_active_profile, suppress_alsa_errors
from neiro.config import Neiro


def record(duration_s: float, cfg: Neiro | None = None) -> np.ndarray:
    """Block for `duration_s` seconds, return float32 mono samples at
    `cfg.audio.input_samplerate` (16 kHz by default).
    """
    cfg = cfg or Neiro()
    suppress_alsa_errors()
    device = resolve_active_profile(cfg)
    apply_to_environment(device)

    sr = cfg.audio.input_samplerate
    sd.check_input_settings(device=device.portaudio_device, samplerate=sr, channels=1)

    frames = int(duration_s * sr)
    audio = sd.rec(
        frames,
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=device.portaudio_device,
        blocksize=cfg.audio.input_blocksize,
        blocking=True,
    )
    return audio.reshape(-1)
