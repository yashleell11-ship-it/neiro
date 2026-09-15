"""Push-to-talk, in the terminal. Stage 0 Task 4.

A deliberately simple dev tool: SPACE toggles recording on and off, 'q'
quits. The system-wide version — a Hyprland global keybind talking to a
persistent daemon over a unix socket — is Task 11, on purpose: a
compositor bind, a socket, and a compiled helper binary are three new
failure modes that have no business standing between "does the loop
work at all" and the very first recording.

Pre-roll: the mic stream is open and feeding a RingBuffer for the whole
life of this process, even while not armed. The instant SPACE is
pressed, the last `pre_roll_s` seconds already sitting in the ring are
pulled out and become the start of the recording — so the first
syllable spoken right as the key goes down is never clipped, which it
would be if capture only began at keydown.

This needs a real interactive terminal (raw mode, byte-at-a-time
keyboard reads) — it cannot be driven non-interactively, which is also
why the actual "does the bar move, is the first word intact" check has
to happen in a real terminal, by ear/eye, same as Task 3's chipmunk test.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
import wave
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live

from elizabeth.audio.devices import (
    apply_to_environment,
    resolve_active_profile,
    suppress_alsa_errors,
)
from elizabeth.audio.ring import RingBuffer
from elizabeth.config import Elizabeth

OUT_PATH = "/tmp/elizabeth-last.wav"
BAR_WIDTH = 40
# Speech RMS on a healthy (non-clipping) mic sits well under 0.25 most of
# the time, so scale the bar by 4x — otherwise normal speech barely
# nudges it and the meter looks broken even when it isn't.
BAR_SCALE = 4.0
METER_SMOOTHING = 0.3  # EMA alpha on the displayed level; pure instantaneous
# RMS flickers too fast at 20fps to read


@dataclass
class _State:
    armed: bool = False
    utterance: list[np.ndarray] = field(default_factory=list, repr=False)
    level: float = 0.0  # smoothed RMS, read by the display loop
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _bar(level: float) -> str:
    filled = max(0, min(BAR_WIDTH, int(level * BAR_WIDTH * BAR_SCALE)))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def _write_wav(audio: np.ndarray, samplerate: int, path: str = OUT_PATH) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        w.writeframes(pcm16.tobytes())


def run(cfg: Elizabeth | None = None) -> int:
    cfg = cfg or Elizabeth()
    console = Console()

    if not sys.stdin.isatty():
        console.print(
            "[red]elizabeth ptt needs a real interactive terminal[/red] (it reads "
            "raw keypresses). Run it directly in a terminal, not piped or "
            "backgrounded."
        )
        return 1

    suppress_alsa_errors()
    device = resolve_active_profile(cfg)
    apply_to_environment(device)

    sr = cfg.audio.input_samplerate
    ring = RingBuffer(capacity_seconds=cfg.audio.ring_buffer_seconds, samplerate=sr)
    state = _State()

    def callback(
        indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        chunk = indata[:, 0].copy()
        ring.write(chunk)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        with state.lock:
            state.level = (1 - METER_SMOOTHING) * state.level + METER_SMOOTHING * rms
            if state.armed:
                state.utterance.append(chunk)

    stream = sd.InputStream(
        device=device.portaudio_device,
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=cfg.audio.input_blocksize,
        callback=callback,
    )

    console.print(
        f"[bold]elizabeth ptt[/bold] — profile '{cfg.audio.active_profile}'. "
        "SPACE to start/stop recording, q to quit.\n"
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    saved_count = 0
    try:
        tty.setraw(fd)
        with stream, Live(console=console, refresh_per_second=20, transient=False) as live:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)

                with state.lock:
                    level = state.level
                    armed = state.armed
                status_text = "[red]● RECORDING[/red]" if armed else "listening"
                live.update(f"{_bar(level)}  {status_text}   (saved: {saved_count})")

                if not ready:
                    continue

                ch = sys.stdin.read(1)
                if ch == "q":
                    break
                if ch != " ":
                    continue

                just_started = False
                utterance_to_save: list[np.ndarray] | None = None
                with state.lock:
                    if not state.armed:
                        preroll = ring.read_last(cfg.audio.pre_roll_s)
                        state.utterance = [preroll] if preroll.size else []
                        state.armed = True
                        just_started = True
                    else:
                        state.armed = False
                        utterance_to_save = state.utterance
                        state.utterance = []

                if just_started:
                    continue

                audio = (
                    np.concatenate(utterance_to_save)
                    if utterance_to_save
                    else np.zeros(0, dtype=np.float32)
                )
                _write_wav(audio, sr)
                saved_count += 1
                peak = float(np.abs(audio).max()) if audio.size else 0.0
                dur = audio.shape[0] / sr if audio.size else 0.0
                silent_note = "  [yellow](looks silent)[/yellow]" if peak < 1e-4 else ""
                console.print(f"\nSaved {OUT_PATH}: {dur:.2f}s, peak {peak:.3f}{silent_note}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    console.print("\nbye.")
    return 0
