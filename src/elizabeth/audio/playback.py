"""Audio playback. Stage 0: a blocking play() for `elizabeth mic-test`'s
chipmunk test. From Stage 2 onward, real TTS playback moves into the
BROWSER's AudioContext (see docs/ARCHITECTURE.md) — this module stays
around only as the dev/debug path and for Stage 0-1's simplest-possible
loop, never as the production sink.

Always 24 kHz mono float32, blocksize 240 (10 ms) when played at the
correct rate — 24 kHz because it's native for every TTS engine this
project considers (Kokoro, Qwen3-TTS, Chatterbox), so nothing downstream
of this module ever needs to resample.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from elizabeth.audio.devices import (
    apply_to_environment,
    resolve_active_profile,
    suppress_alsa_errors,
)
from elizabeth.config import Elizabeth


def play(audio: np.ndarray, samplerate: int, cfg: Elizabeth | None = None) -> None:
    """Play `audio` (float32 mono) at `samplerate`. Blocks until done.

    Deliberately takes `samplerate` as an explicit argument rather than
    always reading `cfg.audio.output_samplerate` — Task 3's chipmunk
    test plays the SAME 16 kHz-captured buffer twice, once honestly at
    16000 and once wrong on purpose at 24000, and that mismatch is the
    entire point: no resampling happens anywhere in this module, so
    telling the device "these 8000 samples are one 24000 Hz second" is
    exactly what makes it sound sped-up and pitch-shifted.
    """
    cfg = cfg or Elizabeth()
    suppress_alsa_errors()
    device = resolve_active_profile(cfg)
    apply_to_environment(device)

    sd.play(audio, samplerate=samplerate, device=device.portaudio_device, blocking=True)
