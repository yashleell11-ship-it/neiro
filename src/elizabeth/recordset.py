"""Guided recording of an evaluation set, in your own voice.

Stage 0 Task 6 (Gate G3a). Prompts for what you're about to say, records
it, saves a numbered WAV plus a reference transcript — so the reference
is captured at record time, while you still remember exactly what you
said, rather than reconstructed later from thirty files.

Raw WAVs stay under data/voice/<name>/ and are gitignored (voice is
biometric — CLAUDE.md rule). Only refs.jsonl, which holds the
transcripts and per-file metadata, is committed.

Deliberate deviation from the plan's sketched `refs.txt`: JSONL instead,
because a flat text file can't carry which device profile recorded each
clip, and Task 3 established that the device matters (earbuds vs the
built-in card are different signal chains, and the arousal baseline in
Gate G3b is per-device). A bare list of sentences would lose that.
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console

from elizabeth.audio.devices import (
    apply_to_environment,
    resolve_active_profile,
    suppress_alsa_errors,
)
from elizabeth.audio.ring import RingBuffer
from elizabeth.config import Elizabeth

DATA_ROOT = Path("data/voice")


def _write_wav(audio: np.ndarray, samplerate: int, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        w.writeframes(pcm16.tobytes())


def run(name: str, count: int, cfg: Elizabeth | None = None) -> int:
    cfg = cfg or Elizabeth()
    console = Console()

    if not sys.stdin.isatty():
        console.print(
            "[red]elizabeth record-set needs a real interactive terminal[/red] — it "
            "prompts you between takes. Run it directly in a terminal."
        )
        return 1

    suppress_alsa_errors()
    device = resolve_active_profile(cfg)
    apply_to_environment(device)

    out_dir = DATA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_path = out_dir / "refs.jsonl"

    # Resume rather than overwrite: re-running after stopping halfway
    # should continue, not silently clobber takes 1..n.
    existing = []
    if refs_path.exists():
        with refs_path.open() as f:
            existing = [json.loads(line) for line in f if line.strip()]
    start_index = len(existing)

    if start_index >= count:
        console.print(
            f"[green]'{name}' already has {start_index} takes[/green] "
            f"(asked for {count}). Delete {refs_path} and the WAVs beside it to "
            "start over, or pass a larger --count to add more."
        )
        return 0

    if start_index:
        console.print(f"[yellow]Resuming[/yellow] — {start_index} takes already recorded.\n")

    sr = cfg.audio.input_samplerate
    ring = RingBuffer(capacity_seconds=cfg.audio.ring_buffer_seconds, samplerate=sr)
    recording: list[np.ndarray] = []
    armed = False

    def callback(
        indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        chunk = indata[:, 0].copy()
        ring.write(chunk)
        if armed:
            recording.append(chunk)

    console.print(
        f"[bold]Recording set '{name}'[/bold] — profile "
        f"'{cfg.audio.active_profile}', {count - start_index} takes to go.\n"
        "For each one: type what you're about to say, press ENTER to start, "
        "speak, then press ENTER again to stop.\n"
        "Type 'q' at any prompt to stop early — takes so far are kept.\n"
    )

    stream = sd.InputStream(
        device=device.portaudio_device,
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=cfg.audio.input_blocksize,
        callback=callback,
    )

    saved = 0
    with stream:
        for i in range(start_index, count):
            console.print(f"[bold cyan]\\[{i + 1}/{count}][/bold cyan]")
            reference = input("  what will you say? ").strip()
            if reference.lower() == "q":
                break
            if not reference:
                console.print("  [yellow]skipped (no reference text)[/yellow]\n")
                continue

            input("  ENTER to start recording... ")
            recording.clear()
            preroll = ring.read_last(cfg.audio.pre_roll_s)
            if preroll.size:
                recording.append(preroll)
            armed = True
            t0 = time.perf_counter()

            input("  recording — ENTER to stop... ")
            armed = False
            duration = time.perf_counter() - t0

            audio = np.concatenate(recording) if recording else np.zeros(0, dtype=np.float32)
            peak = float(np.abs(audio).max()) if audio.size else 0.0

            wav_name = f"{i + 1:03d}.wav"
            _write_wav(audio, sr, out_dir / wav_name)
            with refs_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "file": wav_name,
                            "reference": reference,
                            "profile": cfg.audio.active_profile,
                            "input_device": device.pulse_source,
                            "samplerate": sr,
                            "duration_s": round(audio.shape[0] / sr, 3),
                            "peak": round(peak, 4),
                        }
                    )
                    + "\n"
                )
            saved += 1

            warn = ""
            if peak >= 0.999:
                warn = "  [red]CLIPPING — lower the mic gain[/red]"
            elif peak < 0.01:
                warn = "  [yellow]very quiet — check the mic[/yellow]"
            console.print(f"  saved {wav_name}  {duration:.1f}s  peak {peak:.3f}{warn}\n")

    console.print(f"[green]Done.[/green] {saved} new take(s) in {out_dir}/")
    console.print(f"References: {refs_path}")
    console.print(
        f"\nScore the current STT against them with:  [bold]elizabeth wer --name {name}[/bold]"
    )
    return 0
