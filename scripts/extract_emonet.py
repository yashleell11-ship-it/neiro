#!/usr/bin/env python3
"""Pull EmoNet-Voice Bench's clips out of its parquet shards.

    cd training && uv run python ../scripts/extract_emonet.py

EmoNet-Voice Bench is 12,600 TTS-generated English clips, each made to
express ONE of 42 fine categories and then scored by two to four
psychology experts for how strongly that category is actually heard
(0 / 1 / 2). Audio lives inside the parquet as 44.1 kHz MP3 bytes, which
`corpora.py` cannot index (it walks files), so this writes every clip
out once as 16 kHz mono WAV and records the label beside it in
`extracted/index.csv` — one row per (clip, category), every rater's
score kept verbatim.

Every decodable clip is written, including the ones the experts did not
agree on. Which rows to *train on* is the reader's decision
(`corpora.emonet_agreed`), where it is tested and can change without
decoding 36 hours of MP3 again. What is skipped here is only what can
never be used: a label cell that does not parse, or audio that does not
decode.

Files are named `<source stem>-<sha256[:8]>.wav`. The source `path` is
not a key — 197 of the 12,600 repeat, mostly with different audio — and
the bytes nearly are, so a name from both is stable across runs and
writes a byte-identical clip once.

The index is written last, through a rename, so an interrupted run
leaves no index and the reader sees zero rows rather than a partial
corpus that trains as though it were whole.

Speaker identity is not in the data. The 8-hex prefix of the source
filename is the only identity there is (a few segments of one
generation share it), so it becomes the pseudo-speaker — see the
reader's docstring for why the speaker-independent split is weaker on
this corpus than on any other.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.affect.labels import normalise
from elizabeth.training.corpora import (
    EMONET_INDEX,
    EMONET_INDEX_COLUMNS,
    emonet_agreed,
    parse_emonet_label,
)


def _write_wav(target: Path, blob: bytes, sr_out: int) -> None:
    """Decode one MP3 blob to a 16 kHz mono wav, through a temp name so a
    kill mid-write cannot leave a truncated `.wav` that the next run
    would keep because it exists.
    """
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sr_out:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out)
    tmp = target.with_name(target.name + ".tmp")
    sf.write(tmp, audio, sr_out, format="WAV")
    tmp.replace(target)


def main(argv: list[str] | None = None) -> int:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=REPO / "data/datasets/emonet-voice-bench/data")
    ap.add_argument("--out", type=Path, default=REPO / "data/datasets/emonet-voice-bench/extracted")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--force", action="store_true", help="rebuild the index even if it exists")
    args = ap.parse_args(argv)

    shards = sorted(args.src.glob("*.parquet"))
    if not shards:
        print(
            f"No parquet under {args.src} — run `elizabeth fetch-datasets --only emonet-voice-bench`."
        )
        return 2
    index_path = args.out / EMONET_INDEX
    if index_path.exists() and not args.force:
        print(f"{index_path} already exists — nothing to do (--force rebuilds it)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str, str, str]] = []
    rows_by_category: Counter[str] = Counter()
    agreed_by_category: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    written = 0
    for shard in shards:
        reader = pq.ParquetFile(shard)
        # One row group at a time (~100 clips): a whole shard is 360 MB
        # of MP3 and there is no reason to hold it.
        for group in range(reader.metadata.num_row_groups):
            table = reader.read_row_group(group, columns=["audioId", "label"])
            for row in table.to_pylist():
                parsed = parse_emonet_label(row.get("label") or "")
                if parsed is None:
                    skipped["label"] += 1
                    continue
                category, intensities = parsed
                cell = row.get("audioId") or {}
                blob = cell.get("bytes")
                if not blob:
                    skipped["audio"] += 1
                    continue
                stem = Path(cell.get("path") or "clip").stem
                name = f"{stem}-{hashlib.sha256(blob).hexdigest()[:8]}.wav"
                target = args.out / name
                if not target.exists():
                    try:
                        _write_wav(target, blob, args.sr)
                    except Exception as exc:  # noqa: BLE001 — one bad clip skips, it must not stop the run
                        skipped["audio"] += 1
                        print(f"\n  skipped {stem}: {type(exc).__name__}", file=sys.stderr)
                        continue
                    written += 1
                clip = stem.split("_")[0]
                records.append((name, clip, category, ";".join(str(s) for s in intensities)))
                rows_by_category[category] += 1
                if emonet_agreed(intensities):
                    agreed_by_category[category] += 1
            print(
                f"\r  {shard.name}: {written} clips written, {len(records)} label rows",
                end="",
                file=sys.stderr,
                flush=True,
            )
    print(file=sys.stderr)

    tmp = index_path.with_name(index_path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(EMONET_INDEX_COLUMNS)
        writer.writerows(records)
    tmp.replace(index_path)

    print(f"{written} clips -> {args.out}; {len(records)} label rows -> {index_path.name}")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    print(f"  {'category':<30}{'rows':>6}{'agreed':>8}  -> circumplex")
    for category, n in sorted(rows_by_category.items()):
        point = normalise(category) or "(dropped)"
        print(f"  {category:<30}{n:>6}{agreed_by_category[category]:>8}  -> {point}")
    total, unanimous = sum(rows_by_category.values()), sum(agreed_by_category.values())
    print(f"  {'total':<30}{total:>6}{unanimous:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
