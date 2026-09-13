#!/usr/bin/env python3
"""WER for a recogniser against a corpus with transcripts.

    source env.sh
    uv run scripts/bench_stt.py --corpus svarah --limit 200
    uv run scripts/bench_stt.py --corpus svarah --engine moonshine

Gate G3a's number is WER on **Yash's own voice** — that is the only
accuracy figure that decides anything, and it needs `neiro record-set`
first. This is the thing to run *before* that: a public
Indian-accented-English corpus gives an honest prior for what his accent
costs, without an afternoon of recording.

**Corpus-level WER, not the mean of per-utterance WERs.** They are
different numbers and the mean is the wrong one: a three-word utterance
with one error scores 33%, and averaging that against a thirty-word
utterance with one error (3%) weights the short one ten times too
heavily. Total errors over total reference words is what people mean by
WER.

Per-utterance scores are still reported — sorted worst-first, because
the worst ones are where the useful information is. A WER of 8% made of
uniformly small errors is a different problem from 8% made of three
catastrophic failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.config import Neiro
from neiro.evals.wer import UtteranceScore, edit_distance, normalize, score

SAMPLERATE = 16000

# Corpora that ship transcripts, and where each keeps them. Adding one is
# a line here rather than a change to the harness.
CORPORA = {
    "svarah": {"dir": "svarah-indic-accented-english", "text": "text", "audio": "audio"},
    "fleurs": {"dir": "fleurs", "text": "transcription", "audio": "audio"},
    "common-voice-17-0": {"dir": "common-voice-17-0", "text": "sentence", "audio": "audio"},
    "kathbath": {"dir": "kathbath", "text": "text", "audio": "audio"},
    "indicvoices": {"dir": "indicvoices", "text": "verbatim", "audio": "audio"},
}


def load_rows(corpus: str, limit: int) -> list[tuple[bytes, str]]:
    """`(audio_bytes, reference_text)` from the corpus's parquet shards.

    pyarrow lives in the training venv, not the runtime one, so this
    shells out rather than adding a dependency the daemon would carry
    forever for a benchmark it never runs.
    """
    import subprocess

    spec = CORPORA[corpus]
    code = f'''
import pyarrow.parquet as pq, pickle, sys, glob
shards = sorted(glob.glob("data/datasets/{spec["dir"]}/**/*.parquet", recursive=True))
out = []
for s in shards:
    t = pq.read_table(s)
    names = t.schema.names
    text_col = next((c for c in ("{spec["text"]}", "text", "sentence", "transcription", "verbatim") if c in names), None)
    audio_col = next((c for c in ("{spec["audio"]}", "audio") if c in names), None)
    if not text_col or not audio_col:
        continue
    for r in t.select([audio_col, text_col]).to_pylist():
        a = r[audio_col]
        b = a.get("bytes") if isinstance(a, dict) else None
        if b and r[text_col]:
            out.append((b, str(r[text_col])))
        if len(out) >= {limit}:
            break
    if len(out) >= {limit}:
        break
sys.stdout.buffer.write(pickle.dumps(out))
'''
    raw = subprocess.run(
        ["uv", "run", "--project", "training", "python", "-c", code],
        capture_output=True,
        cwd=REPO,
        check=False,
    ).stdout
    if not raw:
        return []
    import pickle

    return pickle.loads(raw)


def main(argv: list[str] | None = None) -> int:
    import io

    import soundfile as sf

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="svarah")
    ap.add_argument("--engine", choices=["distil", "moonshine"], default="distil")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = load_rows(args.corpus, args.limit)
    if not rows:
        print(f"No transcribed audio found for {args.corpus}. Is it downloaded and complete?")
        return 2
    print(f"{len(rows)} utterances from {args.corpus}")

    cfg = Neiro()
    if args.engine == "distil":
        from neiro.stt.faster_whisper import FasterWhisperStt

        stt = FasterWhisperStt(cfg)
    else:
        from neiro.stt.moonshine import MoonshineStt

        stt = MoonshineStt(cfg)
    warm = stt.warm()
    print(f"warm: {warm:.1f}s")

    import asyncio

    pairs: list[tuple[str, str, str]] = []
    latencies: list[float] = []
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

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else float("nan")
    report = {
        "corpus": args.corpus,
        "engine": args.engine,
        "n": len(pairs),
        # Corpus-level: total errors over total reference words. NOT the
        # mean of per-utterance rates, which weights short utterances
        # ten times too heavily.
        "wer": round(result.wer, 4),
        "total_errors": sum(u.errors for u in per_utterance),
        "total_ref_words": sum(u.ref_words for u in per_utterance),
        "stt_p50_ms": round(p50, 1),
        "worst": [
            {"name": u.name, "ref": u.reference[:90], "hyp": u.hypothesis[:90], "errors": u.errors}
            for u in per_utterance[:5]
        ],
    }
    out = args.out or REPO / "docs" / f"wer-{args.corpus}-{args.engine}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "worst"}, indent=1))
    print("\nworst utterances — where the useful information is:")
    for w in report["worst"]:
        print(f"  {w['errors']:>3} err  ref {w['ref']!r}\n            hyp {w['hyp']!r}")
    print(f"\nWER {result.wer:.1%} on {args.corpus}. Gate G3a is < 10% on HIS voice, not this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
