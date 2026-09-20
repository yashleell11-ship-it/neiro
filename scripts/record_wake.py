#!/usr/bin/env python3
"""Record Yash saying "Hey Elizabeth", because nothing has heard a human say it.

    uv run python scripts/record_wake.py            # 30 of the phrase
    uv run python scripts/record_wake.py --kind negative --count 20

**Why this exists.** The wake head trains on this project's own Kokoro
voice across 28 English voices, and measured on held-out speech it sits
at roughly 70% recall for 0.55 false accepts an hour — you would say her
name 1.4 times on average. Mining the negatives got the false-accept
rate down by most of an order of magnitude and then stopped helping,
which is the shape of a problem that is no longer about the classifier.

Kokoro is a narrow prosody distribution: one synthesiser's idea of how a
sentence is said, at three speeds. A real person says a wake word
differently every time — half-asleep, across the room, mid-sentence,
annoyed that it did not hear them the first time. Thirty recordings of
the real thing are worth more than another thousand synthetic ones, and
no amount of model work substitutes for them.

**Negatives matter as much.** `--kind negative` records ordinary talking
in the same voice, same room, same microphone. The false accepts that
actually cost you are your own voice saying something else, and the
head has never heard that either.

**The audio never leaves this machine.** `data/voice/**/*.wav` is
gitignored (CLAUDE.md rule 7); only the derived embeddings reach the
training set, and even those stay local. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.config import Elizabeth

# Long enough for a slow "Hey Elizabeth" with room either side; the
# trainer's own window is 1.96 s and it needs air around the phrase to
# place it realistically.
CLIP_SECONDS = 2.5
# A beat between the prompt and the recording, so the keystroke is not in
# the clip and you are not still inhaling.
LEAD_IN_S = 0.4


def record_one(seconds: float, samplerate: int, device: str | None) -> np.ndarray:
    import sounddevice as sd

    frames = int(seconds * samplerate)
    audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--kind", choices=["positive", "negative"], default="positive")
    ap.add_argument("--seconds", type=float, default=CLIP_SECONDS)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "voice" / "wake")
    args = ap.parse_args()

    import soundfile as sf

    from elizabeth.audio.devices import (
        apply_to_environment,
        resolve_active_profile,
        suppress_alsa_errors,
    )

    cfg = Elizabeth()
    suppress_alsa_errors()
    device = resolve_active_profile(cfg)
    apply_to_environment(device)
    rate = cfg.audio.input_samplerate

    out = args.out / args.kind
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.wav"))
    start = len(existing)

    phrase = cfg.wake.phrase
    if args.kind == "positive":
        print(f'Say "{phrase}" once per prompt — {args.count} times.')
        print("Vary it on purpose: normal, quiet, from across the room, half-asleep,")
        print("fast, annoyed, mid-sentence. Sameness is what makes a wake word brittle.")
    else:
        print(f"Say ANYTHING EXCEPT \"{phrase}\" — {args.count} times.")
        print("Ordinary sentences, and a few near-misses on purpose: 'hey',")
        print("'Elizabeth', 'hey Eliza', your name for someone else. These are the")
        print("false accepts that will actually cost you.")
    print(f"\n{args.seconds:.1f}s per clip. {start} already recorded in {out}.")
    print("ENTER to record, 'q' then ENTER to stop.\n")

    written = 0
    for i in range(args.count):
        n = start + i
        try:
            key = input(f"[{i + 1}/{args.count}] ENTER to record > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if key.strip().lower() == "q":
            break
        time.sleep(LEAD_IN_S)
        print("   ● recording…", end="", flush=True)
        audio = record_one(args.seconds, rate, device.portaudio_device)
        peak = float(np.max(np.abs(audio)))
        path = out / f"{args.kind}_{n:04d}.wav"
        sf.write(path, audio, rate)
        written += 1
        # Peak is shown every time on purpose: a clipped or silent clip is
        # worse than no clip, and you cannot tell either by ear afterwards.
        flag = "  ⚠ CLIPPED" if peak >= 0.99 else ("  ⚠ very quiet" if peak < 0.02 else "")
        print(f"\r   saved {path.name}  peak {peak:.2f}{flag}      ")

    print(f"\n{written} new clips. Total {args.kind}: {len(list(out.glob('*.wav')))}")
    if written:
        print("These stay on this machine — data/voice/**/*.wav is gitignored.")
        print(f"Play one back:  paplay {out / f'{args.kind}_{start:04d}.wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
