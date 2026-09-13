#!/usr/bin/env python3
"""Pull Rasa's emotion-labelled clips out of its parquet shards.

    cd training && uv run python ../scripts/extract_rasa.py

Rasa is an expressive Indic TTS corpus, so most of it is *reading
styles* — CONV, WIKI, BOOK, NEWS, PROPER NOUN. Only six of its sixteen
styles are emotions, and those are the ~4.5 k clips worth anything to
Lane B. Everything else is skipped rather than mapped to neutral: a
Wikipedia read is not a neutral emotional state, it is a different task.

Audio lives inside the parquet as bytes, which `corpora.py` cannot index
(it walks files). So this writes the emotional clips out once, named
`<speaker>_<STYLE>_<n>.wav`, which the Rasa reader then parses the same
way CREMA-D's filenames are parsed.

Speaker identity is not in the data — Rasa ships one male and one female
voice per language — so gender stands in for speaker, which is what the
speaker-independent split actually needs here.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.affect.labels import normalise

# Rasa's six emotion styles. The other ten are reading registers.
EMOTION_STYLES = {"ANGER", "FEAR", "HAPPY", "SAD", "DISGUST", "SURPRISE"}


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=REPO / "data/datasets/rasa/Hindi")
    ap.add_argument("--out", type=Path, default=REPO / "data/datasets/rasa/extracted")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args(argv)

    shards = sorted(args.src.glob("*.parquet"))
    if not shards:
        print(f"No parquet under {args.src} — run `neiro fetch-datasets --only rasa`.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["filename", "style", "gender", "audio"])
        for row in table.to_pylist():
            style = (row.get("style") or "").strip().upper()
            if style not in EMOTION_STYLES:
                continue
            blob = (row.get("audio") or {}).get("bytes")
            if not blob:
                continue
            try:
                audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
            except Exception as exc:  # noqa: BLE001 — one bad clip skips, it must not stop the run
                print(f"\n  skipped {row.get('filename')}: {type(exc).__name__}", file=sys.stderr)
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != args.sr:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
            speaker = (row.get("gender") or "unknown").strip().lower()
            name = f"{speaker}_{style}_{counts[style]:05d}.wav"
            sf.write(args.out / name, audio, args.sr)
            counts[style] += 1
            written += 1
        print(f"\r  {shard.name}: {written} clips", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    print(f"{written} emotional clips -> {args.out}")
    for style, n in counts.most_common():
        print(f"  {style:<10} {n:>5}  -> {normalise(style)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
