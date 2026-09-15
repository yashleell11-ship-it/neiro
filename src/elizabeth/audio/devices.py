"""Route Elizabeth's audio to a NAMED PipeWire node, never 'default'.

Real finding from Stage 0 Task 3: PortAudio's ALSA host API does not
enumerate PipeWire objects by their pactl names at all. `sd.query_devices()`
on this machine shows generic ALSA PCM aliases — 'pipewire', 'pulse',
'default', plus whatever hardware card happens to be plugged in right
now (e.g. 'Built-in Audio Analog Stereo') — never
'alsa_input.pci-0000_00_1f.3.analog-stereo' or
'bluez_input.11:11:22:33:47:91', which is what Task 0's config.toml
actually stores (see doctor.py's audio_inventory()).

The mechanism that closes this gap: PipeWire's PulseAudio-compatible
server (pipewire-pulse) honours the standard PulseAudio client
environment variables PULSE_SOURCE / PULSE_SINK, read once when a client
connects. So: always open streams against PortAudio device 'pulse', and
set PULSE_SOURCE / PULSE_SINK to the exact node name from config.toml
right before opening the stream. Verified working on this machine
2026-09-11 (see docs/DECISIONS.md) — a stream opened this way returns
real-shaped audio data rather than erroring or silently falling back to
whatever PipeWire's own default happens to be.

Known limitation, not yet re-verified: this was tested with only one
live audio card (the earbuds were disconnected at the time). Re-run the
same check once two devices are live simultaneously to confirm PULSE_SOURCE
actually discriminates between them and isn't just a no-op that happens
to match the single card present. Also: setting PULSE_SOURCE/PULSE_SINK
mutates process-wide environment variables, which is fine for the
short-lived CLI commands Stage 0 builds (mic-test, record-set) but will
need scoping (or a lock) once Stage 2's long-running daemon opens
capture and playback concurrently and might one day need to retarget
one without disturbing the other.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

from elizabeth.config import Elizabeth

# Kept alive for the process lifetime — ctypes.CFUNCTYPE callbacks are
# garbage-collected like anything else, and ALSA calling into a freed
# callback segfaults rather than raising a catchable Python exception.
_alsa_error_handler_ref = None


def suppress_alsa_errors() -> None:
    """Silence alsa-lib's own stderr spam (the classic
    "Unknown PCM cards.pcm.rear" wall of text on a system whose default
    ALSA config references card profiles PipeWire doesn't provide).

    To someone directing Claude Code rather than reading tracebacks, that
    spam is indistinguishable from a crash. It's harmless noise from
    ALSA's own device-probing, not from anything Elizabeth does.
    """
    global _alsa_error_handler_ref
    if _alsa_error_handler_ref is not None:
        return  # already installed

    handler_t = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    )

    def _noop(filename: bytes, line: int, function: bytes, err: int, fmt: bytes) -> None:
        pass

    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    except OSError:
        return  # no ALSA on this system — nothing to suppress

    _alsa_error_handler_ref = handler_t(_noop)
    asound.snd_lib_error_set_handler(_alsa_error_handler_ref)


@dataclass(frozen=True)
class ResolvedDevice:
    """What to actually pass to sounddevice, plus the env vars that make
    'pulse' target a specific node instead of PipeWire's own default.
    """

    portaudio_device: str  # always 'pulse' on this machine — see module docstring
    pulse_source: str | None
    pulse_sink: str | None
    profile_name: str


class NoActiveProfileError(RuntimeError):
    """Raised instead of silently falling back to 'default' — see Task 0."""


def resolve_active_profile(cfg: Elizabeth | None = None) -> ResolvedDevice:
    cfg = cfg or Elizabeth()
    name = cfg.audio.active_profile
    if not name or name not in cfg.audio.profiles:
        raise NoActiveProfileError(
            "No active audio profile configured. Run: elizabeth doctor --audio-inventory, "
            "then set [audio] active_profile and the matching "
            "[audio.profiles.<name>] table in ~/.config/elizabeth/config.toml."
        )
    profile = cfg.audio.profiles[name]
    return ResolvedDevice(
        portaudio_device="pulse",
        pulse_source=profile.input_device,
        pulse_sink=profile.output_device,
        profile_name=name,
    )


def apply_to_environment(device: ResolvedDevice) -> None:
    """Set PULSE_SOURCE/PULSE_SINK for the CURRENT process. Must be
    called before opening the 'pulse' PortAudio device — pipewire-pulse
    reads these once, at client-connect time (i.e. at stream open, not
    at process start — unlike env.sh's LD_LIBRARY_PATH, this is safe to
    set from inside Python).
    """
    if device.pulse_source:
        os.environ["PULSE_SOURCE"] = device.pulse_source
    if device.pulse_sink:
        os.environ["PULSE_SINK"] = device.pulse_sink
