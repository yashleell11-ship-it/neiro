"""Gate G3b's scorer: can we hear how activated he sounds?

This is the number that decides whether the differentiator is real. The
plan's threshold is **arousal AUC > 0.80 on Yash's own voice, on his own
device** — not on a corpus, and not pooled across people.

**Per speaker, always.** A pooled AUC over many speakers measures
whether loud people differ from quiet people, which is not the question.
The question is whether *this* person sounds more activated than *this
person's* own normal. So every speaker is scored against their own
baseline and the AUCs are averaged, and `pooled` is reported alongside
purely to show the gap.

**Ties are averaged.** A model that outputs one constant scores 0.5 here,
which is what it deserves. Ignoring ties would score it 1.0.

**Valence is reported and not gated.** Measured at 0.511 on 30 CREMA-D
speakers — chance. It is printed so the asymmetry stays visible rather
than quietly dropped when it disappoints.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

GO_THRESHOLD = 0.80
MIN_PER_SPEAKER = 6  # below this an AUC is noise, not a measurement


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC with ties averaged.

    Averaging ties is not a detail: without it a constant predictor
    scores 1.0 instead of the 0.5 it has earned, and a broken model
    looks like a perfect one.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
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


@dataclass(frozen=True)
class ArousalScore:
    n_speakers: int
    n_utterances: int
    per_speaker_mean: float
    per_speaker_median: float
    worst_speaker: float
    best_speaker: float
    pooled: float
    valence_mean: float | None

    @property
    def passed(self) -> bool:
        return self.per_speaker_mean > GO_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "gate": "G3b",
            "n_speakers": self.n_speakers,
            "n_utterances": self.n_utterances,
            "arousal_auc_per_speaker_mean": round(self.per_speaker_mean, 4),
            "arousal_auc_per_speaker_median": round(self.per_speaker_median, 4),
            "arousal_auc_worst_speaker": round(self.worst_speaker, 4),
            "arousal_auc_best_speaker": round(self.best_speaker, 4),
            "arousal_auc_pooled": round(self.pooled, 4),
            "valence_auc_per_speaker_mean": (
                round(self.valence_mean, 4) if self.valence_mean is not None else None
            ),
            "threshold": GO_THRESHOLD,
            "verdict": "GO" if self.passed else "NO-GO",
        }


def score(
    rows: list[tuple[str, float, bool]],
    valence_rows: list[tuple[str, float, bool]] | None = None,
) -> ArousalScore:
    """`rows` is `(speaker, arousal_score, is_high_arousal)`.

    The caller decides what "high" means — for acted corpora it is the
    label's position on the circumplex; for Yash's own set it is which
    of the five deliveries he was asked for.
    """
    by_speaker: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for speaker, value, high in rows:
        by_speaker[speaker].append((value, high))

    aucs = []
    for pairs in by_speaker.values():
        if len(pairs) < MIN_PER_SPEAKER:
            continue
        values = np.array([p[0] for p in pairs])
        highs = np.array([p[1] for p in pairs], dtype=bool)
        if highs.all() or (~highs).all():
            continue  # one class only — nothing to separate
        aucs.append(auc(values, highs))

    valence_mean = None
    if valence_rows:
        v_by: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for speaker, value, high in valence_rows:
            v_by[speaker].append((value, high))
        v_aucs = [
            auc(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=bool))
            for pairs in v_by.values()
            if len(pairs) >= MIN_PER_SPEAKER
            and not all(p[1] for p in pairs)
            and any(p[1] for p in pairs)
        ]
        if v_aucs:
            valence_mean = float(np.mean(v_aucs))

    pooled_values = np.array([r[1] for r in rows])
    pooled_high = np.array([r[2] for r in rows], dtype=bool)
    return ArousalScore(
        n_speakers=len(aucs),
        n_utterances=len(rows),
        per_speaker_mean=float(np.mean(aucs)) if aucs else float("nan"),
        per_speaker_median=float(np.median(aucs)) if aucs else float("nan"),
        worst_speaker=float(np.min(aucs)) if aucs else float("nan"),
        best_speaker=float(np.max(aucs)) if aucs else float("nan"),
        pooled=auc(pooled_values, pooled_high),
        valence_mean=valence_mean,
    )
