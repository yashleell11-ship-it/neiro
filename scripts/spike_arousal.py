#!/usr/bin/env python3
"""Can Lane A's prosody separate high arousal from low — and on whose voice?

    uv run scripts/spike_arousal.py                      # on CREMA-D's actors
    uv run scripts/spike_arousal.py --corpora crema-d --per-speaker
    uv run scripts/spike_arousal.py --dir data/voice/emotion   # on Yash's own set

This is the *rehearsal* for gate G3b, not the gate itself. G3b is
measured on Yash's own recordings, on his own device, and the plan's GO
threshold (arousal AUC > 0.80) applies there. Running it on acted
corpora first answers a different and still useful question: **is the
Lane A implementation capable of the separation at all?** If it cannot
tell a professional actor's anger from their sadness — where the
contrast is deliberately exaggerated — then the features or the
baseline are broken, and there is no point recording fifty utterances to
find that out.

Per-speaker mode is the honest version. Each CREMA-D actor gets their
own baseline built from their own neutral clips, which is exactly how it
works at runtime: z-scores against *that person's* normal, never a
pooled average across people.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.affect import features as feat
from elizabeth.affect.baseline import SpeakerBaseline
from elizabeth.affect.labels import CIRCUMPLEX
from elizabeth.affect.prosody import AROUSAL_WEIGHTS, VALENCE_WEIGHTS, _composite
from elizabeth.config import Elizabeth
from elizabeth.training.corpora import Utterance, ensure_extracted, load

SAMPLERATE = 16000


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC with tie handling. A constant predictor scores 0.5, which
    is the whole reason ties are averaged rather than ignored.
    """
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def read_audio(path: str) -> np.ndarray | None:
    import soundfile as sf

    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 — one unreadable clip skips, it does not stop the run
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLERATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLERATE)
    return audio


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpora", nargs="*", default=["crema-d"])
    ap.add_argument("--limit-speakers", type=int, default=20, help="0 for all")
    ap.add_argument("--min-per-speaker", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "spike-arousal.json")
    args = ap.parse_args(argv)

    cfg = Elizabeth()
    print(f"warming librosa: {feat.warm(cfg) * 1000:.0f} ms")

    for name in args.corpora:
        if (REPO / "data" / "datasets" / name).is_dir():
            ensure_extracted(name)
    rows = [r for r in load(args.corpora) if not r.in_archive]
    if not rows:
        print("No audio found. Run `elizabeth fetch-datasets --target ser_lane_b` first.")
        return 2

    by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker].append(row)
    speakers = [s for s, rs in by_speaker.items() if len(rs) >= args.min_per_speaker]
    speakers.sort()
    if args.limit_speakers:
        speakers = speakers[: args.limit_speakers]
    print(
        f"{len(rows)} clips, using {len(speakers)} speakers with >= {args.min_per_speaker} clips each\n"
    )

    per_speaker_auc: list[float] = []
    valence_auc: list[float] = []
    pooled_scores: list[float] = []
    pooled_high: list[bool] = []
    by_label_z: dict[str, list[float]] = defaultdict(list)

    for speaker in speakers:
        clips = by_speaker[speaker]
        # Baseline from this speaker's NEUTRAL clips only — the runtime
        # equivalent is his ordinary day-to-day speech, not a pooled
        # average over everybody.
        neutral = [c for c in clips if c.label == "neutral"]
        rest = [c for c in clips if c.label != "neutral"]
        if len(neutral) < 3 or len(rest) < 6:
            continue

        baseline = SpeakerBaseline(device=speaker, window=cfg.affect.baseline_window)
        for clip in neutral:
            audio = read_audio(clip.path)
            if audio is None:
                continue
            f = feat.extract(audio, cfg)
            if f:
                baseline.observe(f, cfg)
        if baseline.n < 3:
            continue

        scores, highs, vscores, vhighs = [], [], [], []
        for clip in rest:
            audio = read_audio(clip.path)
            if audio is None:
                continue
            f = feat.extract(audio, cfg)
            if f is None:
                continue
            z = baseline.z_scores(f, cfg)
            arousal = _composite(z, AROUSAL_WEIGHTS)
            valence = _composite(z, VALENCE_WEIGHTS)
            if arousal is None:
                continue
            truth = CIRCUMPLEX[clip.label]
            scores.append(arousal)
            highs.append(truth.arousal > 0)
            by_label_z[clip.label].append(arousal)
            if valence is not None:
                vscores.append(valence)
                vhighs.append(truth.valence > 0)
            pooled_scores.append(arousal)
            pooled_high.append(truth.arousal > 0)

        if len(scores) >= 6 and len(set(highs)) == 2:
            per_speaker_auc.append(auc(np.array(scores), np.array(highs)))
        if len(vscores) >= 6 and len(set(vhighs)) == 2:
            valence_auc.append(auc(np.array(vscores), np.array(vhighs)))

    if not per_speaker_auc:
        print("Not enough usable speakers to score.")
        return 1

    arousal_mean = float(np.mean(per_speaker_auc))
    result = {
        "corpora": args.corpora,
        "speakers_scored": len(per_speaker_auc),
        "arousal_auc_per_speaker_mean": round(arousal_mean, 4),
        "arousal_auc_per_speaker_median": round(float(np.median(per_speaker_auc)), 4),
        "arousal_auc_worst_speaker": round(float(np.min(per_speaker_auc)), 4),
        "arousal_auc_best_speaker": round(float(np.max(per_speaker_auc)), 4),
        "valence_auc_per_speaker_mean": round(float(np.mean(valence_auc)), 4)
        if valence_auc
        else None,
        "arousal_auc_pooled": round(auc(np.array(pooled_scores), np.array(pooled_high)), 4),
        "mean_arousal_z_by_label": {
            k: round(float(np.mean(v)), 3)
            for k, v in sorted(by_label_z.items(), key=lambda kv: -np.mean(kv[1]))
        },
        "note": (
            "Acted speech. This rehearses gate G3b's method; G3b itself is measured "
            "on Yash's own voice and device, where the GO threshold (>0.80) applies."
        ),
    }
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

    print(f"\narousal AUC, per-speaker mean: {arousal_mean:.3f}")
    if arousal_mean < 0.65:
        print("  Below 0.65 on ACTED contrast means the implementation cannot do the")
        print("  separation at all — fix Lane A before recording anything.")
    elif arousal_mean < 0.80:
        print("  Works, but not comfortably. Expect worse on natural speech; G3b is the real test.")
    else:
        print(
            "  The implementation can separate arousal on acted speech. Necessary, not sufficient:"
        )
        print("  the published gap between acted and natural is enormous. G3b decides.")
    # Valence is expected to be much weaker. Saying so here keeps the
    # comparison from being quietly dropped when it disappoints.
    if valence_auc:
        print(
            f"valence AUC, per-speaker mean: {float(np.mean(valence_auc)):.3f}  (expected near chance)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
