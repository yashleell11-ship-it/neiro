#!/usr/bin/env python3
"""Fine-tune a speech-emotion model — Lane B.

    cd training
    uv run python recipes/ser_train.py --dry-run          # shape + one forward pass
    uv run python recipes/ser_train.py --epochs 3
    uv run python recipes/ser_train.py --corpora crema-d ravdess --batch 8

**What it predicts.** Two continuous values, valence and arousal, in
[-1, 1] — the same shape as `UserAffect`. Not one of Elizabeth's six VRM
expressions: those are `ElizabethState`, what *she* feels. Keeping the two
apart is CLAUDE.md rule 5, and a regression head is what makes it
structurally impossible to confuse them.

**Why CCC and not MSE.** Concordance correlation coefficient is the
standard metric for dimensional emotion, and it is a better *loss* here
for a specific reason: MSE is minimised by predicting the dataset mean.
On emotion data that gives a model which outputs "neutral, mildly" for
everything and reports a flattering loss. CCC punishes exactly that,
because it rewards covarying with the labels, not sitting near them.

**Why the numbers will look worse than the papers.** Acted corpora
(CREMA-D, RAVDESS) are performed contrast; natural speech is the real
task, and the best published systems reach macro-F1 0.43 there against
65-92 on acted. The evaluation reports acted and natural separately and
refuses to average them into one headline — see `--report`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.affect.labels import CIRCUMPLEX
from elizabeth.evals.arousal_auc import auc
from elizabeth.training.corpora import (
    READERS,
    Utterance,
    ensure_extracted,
    load,
    split,
    summarise,
)
from elizabeth.training.licences import licences_for

ENCODER = REPO / "models" / "w2v-bert-2.0"
OUT_DIR = REPO / "models" / "ser-lane-b"
SAMPLERATE = 16000


# --------------------------------------------------------------------------
# metric


def ccc(pred: np.ndarray, true: np.ndarray) -> float:
    """Concordance correlation coefficient.

    1.0 is perfect agreement, 0.0 is none. Unlike a correlation it also
    punishes a constant offset or a squashed range, which is what catches
    a model that has learned the shape of the labels but always predicts
    them too close to the mean.
    """
    if pred.size < 2:
        return 0.0
    pm, tm = float(pred.mean()), float(true.mean())
    pv, tv = float(pred.var()), float(true.var())
    cov = float(((pred - pm) * (true - tm)).mean())
    denom = pv + tv + (pm - tm) ** 2
    return 0.0 if denom == 0 else 2 * cov / denom


def ccc_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - CCC, per output dimension, averaged.

    Computed in torch so it can be the training objective. See the module
    docstring on why this is not MSE.
    """
    losses = []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], true[:, i]
        pm, tm = p.mean(), t.mean()
        cov = ((p - pm) * (t - tm)).mean()
        denom = p.var(unbiased=False) + t.var(unbiased=False) + (pm - tm) ** 2
        losses.append(1 - (2 * cov / denom.clamp_min(1e-8)))
    return torch.stack(losses).mean()


def auc_above(pred: np.ndarray, true: np.ndarray, boundary: float) -> float | None:
    """Gate G3b's number: can the labels above `boundary` be told from
    the rest?

    The boundary is the caller's and is fixed for a whole run — by
    default the circumplex origin, so "high arousal" means above
    neutral, the same definition `scripts/spike_arousal.py` and
    `evals/arousal_auc.py` already score against. An earlier version
    binarised at the median of whatever subset it was handed, on the
    theory that this balances the classes. It does not: every label is
    one of a dozen discrete circumplex values, so the median snaps to a
    class boundary that moves with the subset's mix. Measured on the
    real split, the crema-d subset's median put `fear` (0.7) on the HIGH
    side while the rasa subset's median put the same label on the LOW
    side — `by_corpus` was publishing answers to different questions
    under one key, beside a comment that compares them across runs.

    Delegates to the repo's one AUC, which averages ties so a constant
    predictor scores 0.5. `None` when the subset holds a single class:
    undefined, not chance.
    """
    value = auc(pred, true > boundary)
    return None if math.isnan(value) else value


# --------------------------------------------------------------------------
# data


def fit_clip(audio: np.ndarray, length: int, start: int | None = None) -> np.ndarray:
    """Pad or crop one waveform to exactly `length` samples.

    This is THE length convention, shared by training and by the runtime
    provider — `affect/ser.py` imports this function rather than
    re-deriving it, exactly as it imports `SerRegressor` rather than
    re-declaring it. Shorter audio gets trailing zeros, the way a
    recorded clip ends; longer audio is centre-cropped unless the caller
    chooses `start` (training does, for a random crop).

    Why it has to be one function: w2v-bert's feature extractor
    normalises each mel bin over the clip's own time axis, so a 2.5 s
    CREMA-D clip trained inside 4 s of audio was normalised against
    1.5 s of padding at the mel floor, and the head then mean-pooled
    over all of it. The runtime used to hand the model its raw 3 s
    window instead — all speech, no padding — a composition the model
    had not seen in a single training example.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > length:
        if start is None:
            start = (len(audio) - length) // 2
        return audio[start : start + length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio


@dataclass
class Clip:
    audio: np.ndarray
    valence: float
    arousal: float


class EmotionClips(Dataset):
    """Audio → fixed-length float32 at 16 kHz, plus its two targets.

    `seconds` is the clip length the model is trained at. It has no
    default here on purpose: it comes from `--seconds`, is saved into the
    checkpoint with the other arguments, and `affect/ser.py` reads it
    back from there to fit the live window with the same `fit_clip` —
    one number, recorded once, honoured at both ends.
    """

    def __init__(self, rows: list[Utterance], seconds: float, train: bool = False) -> None:
        self.rows = [r for r in rows if not r.in_archive]
        self.length = int(seconds * SAMPLERATE)
        self.train = train
        if len(self.rows) < len(rows):
            print(
                f"  {len(rows) - len(self.rows)} rows are still inside an archive — "
                "run ensure_extracted() for those corpora",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import soundfile as sf

        row = self.rows[index]
        audio, sr = sf.read(row.path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLERATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLERATE)

        start = None
        if self.train and len(audio) > self.length:
            # Train: random crop, for a little augmentation. Eval and
            # runtime: centre, so a number is reproducible run to run.
            start = int(np.random.randint(0, len(audio) - self.length))
        audio = fit_clip(audio, self.length, start=start)

        return (
            torch.from_numpy(np.ascontiguousarray(audio)),
            torch.tensor([row.valence, row.arousal], dtype=torch.float32),
        )


# --------------------------------------------------------------------------
# model


class SerRegressor(nn.Module):
    """A frozen-ish speech encoder plus a two-output head.

    Only the top `unfreeze_layers` encoder blocks are trained. The lower
    layers of a self-supervised speech model encode phonetics, which is
    not what changes between an angry and a sad reading of the same
    sentence — and with ~7k training clips, unfreezing 580M parameters
    would memorise the actors instead.
    """

    def __init__(self, encoder_path: Path, unfreeze_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(str(encoder_path))
        hidden = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = getattr(self.encoder, "encoder", None)
        layers = getattr(blocks, "layers", []) if blocks is not None else []
        for layer in list(layers)[-unfreeze_layers:] if unfreeze_layers else []:
            for param in layer.parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            # The circumplex is defined on [-1, 1]; tanh makes that a
            # property of the model rather than a hope about the loss.
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_features=features).last_hidden_state
        return self.head(out.mean(dim=1))


def make_collate(processor):
    def collate(batch):
        audio = [b[0].numpy() for b in batch]
        targets = torch.stack([b[1] for b in batch])
        feats = processor(audio, sampling_rate=SAMPLERATE, return_tensors="pt")
        return feats["input_features"], targets

    return collate


# --------------------------------------------------------------------------
# loops


def _round4(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


@torch.no_grad()
def evaluate(
    model, loader, device, dtype, arousal_boundary: float, valence_boundary: float
) -> dict:
    """Score one loader. The two boundaries are the run's, not the
    loader's — every subset `main()` reports is scored against the same
    class definition, which is the only way `by_corpus` rows can be
    read side by side.
    """
    model.eval()
    preds, trues = [], []
    for feats, targets in loader:
        out = model(feats.to(device, dtype=dtype))
        preds.append(out.float().cpu().numpy())
        trues.append(targets.numpy())
    if not preds:
        return {"n": 0}
    pred, true = np.concatenate(preds), np.concatenate(trues)
    return {
        "n": len(pred),
        "ccc_valence": round(ccc(pred[:, 0], true[:, 0]), 4),
        "ccc_arousal": round(ccc(pred[:, 1], true[:, 1]), 4),
        "arousal_auc": _round4(auc_above(pred[:, 1], true[:, 1], arousal_boundary)),
        "valence_auc": _round4(auc_above(pred[:, 0], true[:, 0], valence_boundary)),
        "pred_std_arousal": round(float(pred[:, 1].std()), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    """Every argument is saved into the checkpoint and the report, so a
    number in either can always be traced to the run that produced it.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpora", nargs="*", default=None, choices=list(READERS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="clip length every example is fitted to; saved in the checkpoint, and "
        "affect/ser.py fits the live window to that same value",
    )
    ap.add_argument("--unfreeze", type=int, default=4, help="top encoder blocks to train")
    # "High" means above neutral, the circumplex origin — one definition
    # for the whole run and for every subset in the report. See
    # `auc_above` for what happened when each subset chose its own.
    ap.add_argument(
        "--arousal-boundary",
        type=float,
        default=CIRCUMPLEX["neutral"].arousal,
        help="arousal_auc's positive class is label arousal strictly above this",
    )
    ap.add_argument(
        "--valence-boundary",
        type=float,
        default=CIRCUMPLEX["neutral"].valence,
        help="valence_auc's positive class is label valence strictly above this",
    )
    ap.add_argument("--encoder", type=Path, default=ENCODER)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="index, build, one forward pass, stop")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--publishable-only",
        action="store_true",
        help="refuse to train on any corpus whose weights_publishable is not yes",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    boundaries = {
        "arousal_boundary": args.arousal_boundary,
        "valence_boundary": args.valence_boundary,
    }

    for name in args.corpora or list(READERS):
        if (REPO / "data" / "datasets" / name).is_dir():
            ensure_extracted(name)

    rows = load(args.corpora)
    if not rows:
        print(
            "No labelled utterances found. Run `elizabeth fetch-datasets --target ser_lane_b` first."
        )
        return 2
    print(json.dumps(summarise(rows), indent=1))

    # What may be done with the weights this run is about to produce.
    # Decided from the corpora actually indexed, not from what was asked
    # for: a corpus that was requested but found nothing on disk must not
    # restrict a checkpoint it did not contribute to.
    licences = licences_for(sorted({r.corpus for r in rows}))
    restricted = [
        name
        for name, info in licences["per_corpus"].items()
        if info["weights_publishable"] != "yes"
    ]
    if restricted and args.publishable_only:
        print(
            "Refusing to train: --publishable-only, and these corpora do not "
            f"permit publishable weights: {', '.join(restricted)}.\n"
            "Drop them with --corpora, or drop the flag and keep the checkpoint private."
        )
        return 2
    if restricted:
        print(
            f"NOTE: weights from this run are {licences['weights_publishable']} "
            f"for publication ({', '.join(restricted)}). models/ is gitignored; "
            "this is recorded in the checkpoint and in report.json."
        )

    train_rows, val_rows, test_rows = split(rows)
    print(f"split by speaker: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    overlap = {r.speaker for r in train_rows} & {r.speaker for r in test_rows}
    assert not overlap, f"speaker leak: {sorted(overlap)[:5]}"
    # A speaker-hash split can hand a small corpus an empty val or test
    # partition — and then training runs to completion, saves no
    # checkpoint (nothing ever beats the initial best), and reports NaN
    # as though it were a result.
    if not val_rows or not test_rows:
        print(
            f"Split left val={len(val_rows)} test={len(test_rows)}. The speaker hash "
            "put every speaker on one side, which happens with few speakers. Add a "
            "corpus or widen --corpora; training now would save no checkpoint and "
            "report NaN.",
            file=sys.stderr,
        )
        return 2

    if not args.encoder.exists():
        print(f"Encoder not at {args.encoder} — run `elizabeth fetch-models --only w2v-bert-2.0`.")
        return 2

    from transformers import AutoFeatureExtractor

    processor = AutoFeatureExtractor.from_pretrained(str(args.encoder))
    collate = make_collate(processor)

    # Refuse a CPU build rather than training 40x slower and calling the
    # result a number. This is not hypothetical: adding CPU torch to the
    # runtime project silently replaced this venv's CUDA build through
    # the editable path dependency, and the only symptom was an import
    # error deep inside transformers hours later.
    if not torch.cuda.is_available():
        print(
            f"torch {torch.__version__} cannot see a GPU. Training on CPU would take "
            "days and the number would not be comparable. If this says '+cpu', the "
            "runtime project's CPU torch source has leaked in through the editable "
            "path dependency — run `uv sync` in training/ to restore the cu130 build.",
            file=sys.stderr,
        )
        if not args.dry_run:
            return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(
        f"device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}), dtype {dtype}"
    )

    model = SerRegressor(args.encoder, unfreeze_layers=args.unfreeze).to(device, dtype=dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M")

    loaders = {
        name: DataLoader(
            EmotionClips(rs, args.seconds, train=(name == "train")),
            batch_size=args.batch,
            shuffle=(name == "train"),
            num_workers=args.workers,
            collate_fn=collate,
            drop_last=(name == "train"),
        )
        for name, rs in (("train", train_rows), ("val", val_rows), ("test", test_rows))
    }

    if args.dry_run:
        feats, targets = next(iter(loaders["val"]))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(feats.to(device, dtype=dtype))
        print(
            f"forward pass ok: in {tuple(feats.shape)} -> out {tuple(out.shape)} in {(time.perf_counter() - t0) * 1000:.0f} ms"
        )
        print(f"targets {tuple(targets.shape)}, loss {ccc_loss(out.float().cpu(), targets):.4f}")
        if device.type == "cuda":
            print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return 0

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    steps = max(1, len(loaders["train"]) * args.epochs)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps)

    best = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (feats, targets) in enumerate(loaders["train"], 1):
            out = model(feats.to(device, dtype=dtype))
            loss = ccc_loss(out.float(), targets.to(device).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad(set_to_none=True)
            running += float(loss) * len(targets)
            seen += len(targets)
            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(loaders['train'])} loss {running / seen:.4f}"
                )

        scores = evaluate(model, loaders["val"], device, dtype, **boundaries)
        print(f"epoch {epoch}: train_loss {running / max(seen, 1):.4f}  val {json.dumps(scores)}")
        if scores.get("ccc_arousal", -1) > best:
            best = scores["ccc_arousal"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args) | {"encoder": str(args.encoder), "out": str(args.out)},
                    # Travels with the weights, so a checkpoint found on
                    # disk a month later still says what it may be used for.
                    "licences": licences,
                },
                args.out / "best.pt",
            )
            print(f"  saved (best val ccc_arousal {best:.4f})")

    # Reload the BEST checkpoint before scoring the test set.
    #
    # Without this the report describes the model as it stood after the
    # LAST epoch, while `best.pt` — the file `affect/ser.py` actually
    # loads at runtime — holds different weights. Every number published
    # would then be for a model nobody runs, and the gap grows with
    # exactly the overfitting that makes early stopping worth having.
    checkpoint = args.out / "best.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device, dtype=dtype)
        print(f"reloaded {checkpoint.name} (val ccc_arousal {best:.4f}) for the test pass")
    else:
        print("WARNING: no best.pt was saved; reporting the last-epoch model instead")

    final = evaluate(model, loaders["test"], device, dtype, **boundaries)
    by_kind: dict[str, dict] = {}
    for kind, wanted in (("acted", True), ("natural", False)):
        subset = [r for r in test_rows if r.acted is wanted]
        if subset:
            loader = DataLoader(
                EmotionClips(subset, args.seconds),
                batch_size=args.batch,
                num_workers=args.workers,
                collate_fn=collate,
            )
            by_kind[kind] = evaluate(model, loader, device, dtype, **boundaries)

    # Per corpus, because one corpus can dominate a split and make the
    # headline describe something else entirely. Rasa ships exactly TWO
    # speaker ids (male, female), so a speaker-independent split puts all
    # 4541 of its clips on one side — the first run with Rasa included
    # had a test set that was 68% a single unseen Hindi male voice, and
    # the headline AUC dropped from 0.788 to 0.682 while validation ROSE
    # to 0.798. Neither number was wrong; the headline just stopped
    # meaning what it meant the day before.
    by_corpus: dict[str, dict] = {}
    for corpus in sorted({r.corpus for r in test_rows}):
        subset = [r for r in test_rows if r.corpus == corpus]
        if len(subset) < 32:
            continue
        loader = DataLoader(
            EmotionClips(subset, args.seconds),
            batch_size=args.batch,
            num_workers=args.workers,
            collate_fn=collate,
        )
        by_corpus[corpus] = evaluate(model, loader, device, dtype, **boundaries)

    report = {
        "test": final,
        "by_corpus": by_corpus,
        "split_speakers": {
            "train": len({r.speaker for r in train_rows}),
            "val": len({r.speaker for r in val_rows}),
            "test": len({r.speaker for r in test_rows}),
        },
        # Never averaged into one headline: acted contrast and natural
        # speech are different tasks, and reporting one number over a mix
        # of them is the commonest way SER results mislead.
        "by_speech_kind": by_kind,
        "corpora": summarise(rows)["by_corpus"],
        "licences": licences,
        "best_val_ccc_arousal": round(best, 4),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(
        f"\nGate G3b reads `arousal_auc`: {final.get('arousal_auc')} (GO is > 0.80, on HIS voice, not this)"
    )
    print(f"weights publishable: {licences['weights_publishable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
