"""Acoustic echo cancellation — only for the speakers profile.

**Earbuds do not need this and that is the whole design.** With
headphones her voice never reaches the microphone, so barge-in is
trivially reliable and no cancellation is involved. Every convincing
barge-in demo you have seen was recorded on headphones. Speakers are the
honest, harder case, and this is best-effort there.

**The hard part is not the canceller, it is the mic gain.** This laptop
ships Capture at +30 dB with Internal Mic Boost on, which saturates the
input — measured 2026-09-11 at RMS 0.9-0.95 for an entire 3-second clip.
A canceller subtracts a reference from the microphone signal, and you
cannot subtract anything from a clipped waveform because the information
is already gone. So `check_gain()` runs first and says so.

**The SER branch always reads the RAW microphone.** AEC's noise
suppression destroys exactly the prosody features Lane A measures —
energy envelope, pitch continuity — so affect is fed the untouched
signal even when the canceller is active. Getting this backwards would
make her emotionally blind precisely when speakers are in use.

This module does not implement cancellation. PipeWire's
`module-echo-cancel` with WebRTC AEC3 is already on disk; what is
missing is the drop-in and the verification, and writing a worse
canceller in Python would be strictly negative. So: check the
preconditions, print the config, measure the result.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".config/pipewire/pipewire.conf.d/99-neiro-aec.conf"

# Only these six webrtc.* keys are valid in this PipeWire version;
# anything else is silently ignored, which looks like a canceller that
# does not work rather than a typo.
#
# gain_control is OFF deliberately: it fights the level the canceller is
# trying to match, and the plan found it makes things worse.
#
# suspend-timeout 0 on Neiro's nodes matters because PipeWire suspends
# idle nodes after 5 s, and a canceller that has just resumed has not
# adapted — so the first thing she says after a pause echoes.
DROP_IN = """# Neiro: echo cancellation for the SPEAKERS profile.
# Written by `neiro doctor`; never installed silently.
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      library.name = aec/libspa-aec-webrtc
      node.latency = 1024/48000
      capture.props  = { node.name = "neiro_aec_capture"  node.passive = true }
      source.props   = { node.name = "neiro_aec_source"   session.suspend-timeout-seconds = 0 }
      sink.props     = { node.name = "neiro_aec_sink"     session.suspend-timeout-seconds = 0 }
      playback.props = { node.name = "neiro_aec_playback" node.passive = true }
      aec.args = {
        webrtc.gain_control = false
        webrtc.noise_suppression = true
        webrtc.high_pass_filter = true
        webrtc.echo_canceller = true
        webrtc.voice_detection = false
        webrtc.extended_filter = true
      }
    }
  }
]
"""

# Above this fraction of samples at full scale, the input is saturated
# and no canceller can help — the information is already gone.
CLIPPED_FRACTION = 0.01


@dataclass(frozen=True)
class GainCheck:
    peak: float
    rms: float
    clipped_fraction: float

    @property
    def saturated(self) -> bool:
        return self.clipped_fraction > CLIPPED_FRACTION

    def remedy(self) -> str | None:
        if not self.saturated:
            return None
        return (
            f"Input is saturated ({self.clipped_fraction:.0%} of samples at full scale, "
            f"RMS {self.rms:.2f}). A canceller subtracts a reference from the mic "
            "signal and cannot recover a clipped waveform. Lower the gain first:\n"
            "  amixer -c0 sset Capture 50%\n"
            "  amixer -c0 sset 'Internal Mic Boost' 0"
        )


def check_gain(pcm: np.ndarray) -> GainCheck:
    """Is the microphone clipping? Run this BEFORE blaming the canceller."""
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return GainCheck(0.0, 0.0, 0.0)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return GainCheck(peak=peak, rms=rms, clipped_fraction=clipped)


def suppression_db(raw: np.ndarray, cancelled: np.ndarray) -> float:
    """How much quieter the cancelled source is than the raw mic, in dB.

    The acceptance bar is **>15 dB while TTS is playing**. Measured on
    the same audio window through both paths, or the number means
    nothing.
    """
    raw_rms = float(np.sqrt(np.mean(np.asarray(raw, dtype=np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(np.asarray(cancelled, dtype=np.float64) ** 2)))
    if raw_rms <= 0 or out_rms <= 0:
        return float("inf") if raw_rms > 0 else 0.0
    return float(20.0 * np.log10(raw_rms / out_rms))


def module_loaded() -> bool:
    """Is PipeWire's echo-cancel module actually running?

    Installed-but-not-loaded looks exactly like a canceller that does
    not work, so this is checked rather than assumed.
    """
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "modules"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout
        return "echo-cancel" in out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def print_config() -> str:
    """Return the drop-in for a human to install.

    Never written automatically: this changes how every application on
    the machine records audio, which is not a decision a voice assistant
    gets to make on its own.
    """
    return f"# Save as {CONFIG_PATH}, then: systemctl --user restart pipewire\n\n{DROP_IN}"
