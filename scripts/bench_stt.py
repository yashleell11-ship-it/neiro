#!/usr/bin/env python3
"""WER for a recogniser against a corpus with transcripts.

    source env.sh
    uv run scripts/bench_stt.py --corpus svarah --limit 200
    uv run scripts/bench_stt.py --corpus kathbath --language hi --limit 120
    uv run scripts/bench_stt.py --corpus kathbath --language auto --limit 120
    uv run scripts/bench_stt.py --corpus svarah --engine moonshine

Gate G3a's number is WER on **Yash's own voice** — that is the only
accuracy figure that decides anything, and it needs `neiro record-set`
first. This is the thing to run *before* that: a public
Indian-accented-English corpus gives an honest prior for what his accent
costs, without an afternoon of recording — and the two Hindi corpora
give the same prior for the other half of what he speaks.

**Hindi is measured the same way as English, on purpose.** `--language`
maps onto `cfg.stt.language`. "auto" is what the daemon actually runs,
so its number is the honest one; pinning "hi" separates the
recogniser's Hindi from its language detector's, and the gap between
the two runs is what detection costs. The report also carries the
detector's histogram, because "Hindi scored badly" and "Hindi was
tagged as Urdu" are different problems.

**Corpus-level WER, not the mean of per-utterance WERs.** They are
different numbers and the mean is the wrong one: a three-word utterance
with one error scores 33%, and averaging that against a thirty-word
utterance with one error (3%) weights the short one ten times too
heavily. Total errors over total reference words is what people mean by
WER. Latency is p50/p95 by nearest rank for the same family of reason —
a mean hides the one slow utterance that makes her feel broken.

Per-utterance scores are still reported — sorted worst-first, because
the worst ones are where the useful information is. A WER of 8% made of
uniformly small errors is a different problem from 8% made of three
catastrophic failures.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.config import Neiro
from neiro.evals.latency import percentile
from neiro.evals.wer import UtteranceScore, edit_distance, normalize, score

SAMPLERATE = 16000
DATASETS_DIR = REPO / "data" / "datasets"
# The marker scripts/fetch_datasets.py leaves when a download finished.
# Its absence is not fatal here — a half-downloaded corpus still yields
# rows — but it is worth one line on stderr, because a WER on a partial
# corpus is a WER on whichever shards happened to arrive first.
COMPLETE_MARKER = ".neiro-complete"
LANGUAGES = ("auto", "en", "hi")

# Corpora that ship transcripts, and where each keeps them. Adding one is
# a line here rather than a change to the harness.
#
# `split` names the held-out shards to prefer. Both Hindi corpora are
# also fine-tuning sources (target `hindi_stt` in data/datasets.toml), so
# scoring their train rows would let a later fine-tune mark its own
# homework. When the split is absent the harness falls back to whatever
# shards exist, and says so.
CORPORA = {
    # Svarah keeps the audio bytes under `audio_filepath`, not `audio` —
    # the name says path and the value is a {bytes, path} dict. Guessing
    # "audio" found nothing and reported "is it downloaded?", which is
    # the wrong diagnosis for a column-name mismatch. Kathbath and
    # IndicVoices use the same column name (checked 2026-09-14).
    "svarah": {
        "dir": "svarah-indic-accented-english",
        "text": "text",
        "audio": "audio_filepath",
        "split": "test",
    },
    "fleurs": {"dir": "fleurs", "text": "transcription", "audio": "audio"},
    "common-voice-17-0": {"dir": "common-voice-17-0", "text": "sentence", "audio": "audio"},
    "kathbath": {"dir": "kathbath", "text": "text", "audio": "audio_filepath", "split": "valid"},
    # `verbatim` is what the speaker said, mispronunciations included;
    # `normalized` is the corrected text. A recogniser is asked what was
    # said, so it is scored against verbatim.
    "indicvoices": {
        "dir": "indicvoices",
        "text": "verbatim",
        "audio": "audio_filepath",
        "split": "valid",
    },
}

# Column names tried, in order, when a corpus's own is absent from a
# shard — so a schema drift degrades to a different column, visibly in
# the report, rather than to an empty reference set.
TEXT_FALLBACKS = ("text", "sentence", "transcription", "verbatim")
AUDIO_FALLBACKS = ("audio", "audio_filepath")

# Runs inside the interpreter that has pyarrow (see `load_rows`). Reads a
# JSON job from stdin, writes a pickle to stdout. One unreadable shard —
# truncated by a download that stopped, deleted since the glob ran — is
# reported and skipped, never fatal: the run is about the recogniser,
# not the disk.
_READER = r"""
import json, pickle, sys
import pyarrow.parquet as pq

job = json.load(sys.stdin)
rows, skipped = [], []
for shard in job["shards"]:
    try:
        pf = pq.ParquetFile(shard)
    except Exception as exc:
        skipped.append(f"{shard}: {type(exc).__name__}")
        continue
    names = pf.schema_arrow.names
    text_col = next((c for c in job["text_cols"] if c in names), None)
    audio_col = next((c for c in job["audio_cols"] if c in names), None)
    if not text_col or not audio_col:
        skipped.append(f"{shard}: no transcript/audio column in {names}")
        continue
    try:
        for batch in pf.iter_batches(columns=[audio_col, text_col]):
            for r in batch.to_pylist():
                a = r[audio_col]
                b = a.get("bytes") if isinstance(a, dict) else None
                if b and r[text_col]:
                    rows.append((b, str(r[text_col])))
                if len(rows) >= job["limit"]:
                    break
            if len(rows) >= job["limit"]:
                break
    except Exception as exc:
        skipped.append(f"{shard}: {type(exc).__name__} mid-read")
    if len(rows) >= job["limit"]:
        break
sys.stdout.buffer.write(pickle.dumps({"rows": rows, "skipped": skipped}))
"""


def select_shards(root: Path, spec: dict) -> list[Path]:
    """The parquet shards to read, in a stable order, held-out split first.

    Hidden directories are excluded: HF's `.cache/` keeps lock and
    partial files next to real shards, and a `.parquet.incomplete` is
    exactly the file that must not be read.
    """
    split = spec.get("split")
    candidates = sorted(root.rglob(f"{split}-*.parquet")) if split else []
    if not candidates:
        candidates = sorted(root.rglob("*.parquet"))
    return [p for p in candidates if not any(part.startswith(".") for part in p.parts)]


def load_rows(
    corpus: str,
    limit: int,
    data_root: Path = DATASETS_DIR,
    reader_python: Path | None = None,
) -> tuple[list[tuple[bytes, str]], list[str]]:
    """`(audio_bytes, reference_text)` rows from the corpus's parquet
    shards, plus a note per shard that could not be used.

    pyarrow lives in the training venv, not the runtime one, so this
    shells out rather than adding a dependency the daemon would carry
    forever for a benchmark it never runs. `reader_python` names an
    interpreter that has pyarrow when `uv run --project training` is not
    the right one — a worktree without its own training venv, say.
    """
    spec = CORPORA[corpus]
    root = data_root / spec["dir"]
    if not root.is_dir():
        return [], [f"{root}: not downloaded"]
    if not (root / COMPLETE_MARKER).exists():
        print(
            f"  {root.name}: no {COMPLETE_MARKER} marker — scoring a partial download",
            file=sys.stderr,
        )
    shards = select_shards(root, spec)
    if not shards:
        return [], [f"{root}: no parquet shards"]
    job = {
        "shards": [str(s) for s in shards],
        "text_cols": [spec["text"], *TEXT_FALLBACKS],
        "audio_cols": [spec["audio"], *AUDIO_FALLBACKS],
        "limit": limit,
    }
    interpreter = (
        [str(reader_python)] if reader_python else ["uv", "run", "--project", "training", "python"]
    )
    proc = subprocess.run(
        [*interpreter, "-c", _READER],
        input=json.dumps(job).encode(),
        capture_output=True,
        cwd=REPO,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        return [], [f"reader exited {proc.returncode}"]
    out = pickle.loads(proc.stdout)
    return out["rows"], out["skipped"]


def override_language(cfg: Neiro, language: str | None) -> Neiro:
    """`--language` wins over config.toml for this run only; None means
    whatever the config says, which is what the daemon would do.
    """
    if language is not None:
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, not {language!r}")
        cfg.stt.language = language
    return cfg


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """p50 / p95 by nearest rank — the one helper, no mean. See
    evals/latency.py for why a mean is deliberately absent.
    """
    return {
        "stt_p50_ms": round(percentile(latencies_ms, 0.50), 1),
        "stt_p95_ms": round(percentile(latencies_ms, 0.95), 1),
    }


def main(argv: list[str] | None = None) -> int:
    import io

    import soundfile as sf

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="svarah")
    ap.add_argument("--engine", choices=["distil", "moonshine"], default="distil")
    ap.add_argument(
        "--language",
        choices=LANGUAGES,
        default=None,
        help="overrides cfg.stt.language for this run; default: whatever config.toml says",
    )
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    ap.add_argument(
        "--reader-python",
        type=Path,
        default=None,
        help="interpreter that has pyarrow; default: the training venv via uv",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows, skipped = load_rows(args.corpus, args.limit, args.data_root, args.reader_python)
    for note in skipped:
        print(f"  skipped shard: {note}", file=sys.stderr)
    if not rows:
        print(f"No transcribed audio found for {args.corpus}. Is it downloaded and complete?")
        return 2
    print(f"{len(rows)} utterances from {args.corpus}")

    cfg = override_language(Neiro(), args.language)
    if args.engine == "distil":
        from neiro.stt.faster_whisper import FasterWhisperStt

        stt = FasterWhisperStt(cfg)
    else:
        from neiro.stt.moonshine import MoonshineStt

        stt = MoonshineStt(cfg)
    print(f"language: {cfg.stt.language}")
    warm = stt.warm()
    print(f"warm: {warm:.1f}s")

    import asyncio

    pairs: list[tuple[str, str, str]] = []
    latencies: list[float] = []
    detected: Counter[str] = Counter()
    for i, (blob, reference) in enumerate(rows, 1):
        try:
            audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001 — one unreadable clip skips the row
            print(f"\n  skipped utterance {i}: {type(exc).__name__}", file=sys.stderr)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLERATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLERATE)
        started = time.perf_counter()
        hypothesis = asyncio.run(stt.transcribe(audio))
        latencies.append((time.perf_counter() - started) * 1000)
        pairs.append((f"{args.corpus}-{i:04d}", reference, hypothesis))
        language = getattr(stt, "last_detected_language", None)
        if language:
            detected[language] += 1
        print(f"\r  {i}/{len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    result = score(pairs)
    per_utterance = sorted(
        (
            UtteranceScore(
                name=n,
                reference=r,
                hypothesis=h,
                errors=edit_distance(normalize(r), normalize(h)),
                ref_words=len(normalize(r)),
            )
            for n, r, h in pairs
        ),
        key=lambda u: -(u.errors / max(1, u.ref_words)),
    )

    report = {
        "corpus": args.corpus,
        "engine": args.engine,
        "language": cfg.stt.language,
        "n": len(pairs),
        # Corpus-level: total errors over total reference words. NOT the
        # mean of per-utterance rates, which weights short utterances
        # ten times too heavily.
        "wer": round(result.wer, 4),
        "total_errors": sum(u.errors for u in per_utterance),
        "total_ref_words": sum(u.ref_words for u in per_utterance),
        # Utterances the provider's confidence floor turned into "" —
        # every reference word a deletion. Two recognisers at the same
        # WER can differ entirely here, and the difference is what he
        # hears: one says nothing, the other says something wrong.
        "rejected": sum(1 for _, _, h in pairs if not h),
        **latency_summary(latencies),
        # What the detector decided, per utterance. Under a pinned
        # language this is trivially all that language; under "auto" it
        # is the number that explains a bad Hindi WER — or exonerates it.
        "detected_languages": dict(detected.most_common()),
        "skipped_shards": skipped,
        "worst": [
            {"name": u.name, "ref": u.reference[:90], "hyp": u.hypothesis[:90], "errors": u.errors}
            for u in per_utterance[:5]
        ],
    }
    out = args.out or REPO / "docs" / f"wer-{args.corpus}-{args.engine}-{cfg.stt.language}.json"
    # ensure_ascii=False so a Devanagari `worst` list is readable in the
    # file, not a wall of \u escapes.
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "worst"}, indent=1, ensure_ascii=False)
    )
    print("\nworst utterances — where the useful information is:")
    for w in report["worst"]:
        print(f"  {w['errors']:>3} err  ref {w['ref']!r}\n            hyp {w['hyp']!r}")
    print(
        f"\nWER {result.wer:.1%} on {args.corpus} (language={cfg.stt.language}). "
        "Gate G3a is < 10% on HIS voice, not this."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
