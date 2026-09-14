#!/usr/bin/env python3
"""Fine-tune Whisper large-v3-turbo on Hindi — Kathbath + IndicVoices.

    cd training
    uv run python recipes/stt_train.py --dry-run            # shapes, params, peak VRAM
    uv run python recipes/stt_train.py --eval-only          # the BEFORE number
    uv run python recipes/stt_train.py --max-hours 8 --batch 4 --accum 8

**Why this run exists.** The same checkpoint, on the same machine, in
the same week: 6.2% WER on Svarah (Indian-accented English) and **29.6%
on Kathbath Hindi** — `docs/svarah-auto-turbo.json` and
`docs/kathbath-hi-turbo.json`. Neiro is meant to answer in both
languages, and a recogniser that is five times worse in one of them is
not bilingual, it is an English assistant that tolerates Hindi. The
worst Kathbath utterances are not near-misses either: they are the model
dropping into Latin script mid-sentence ("B.S.C. के Mid Camp और Small
Camp"), which is a recogniser that has decided the speaker is code-
switching when they are not.

**Why LoRA and not a full fine-tune.** 8151 MiB total, ~7730 usable, and
the other half of the box is often busy. A full fine-tune of 809M
parameters needs the weights (1.6 GB in bf16), their gradients (1.6 GB)
and AdamW's two moments in fp32 (6.5 GB) — 9.7 GB before a single 30 s
mel spectrogram. LoRA on the attention projections trains 3.28M of
them — 0.40% — so the optimiser state is megabytes and the ceiling
becomes activation memory, which gradient checkpointing and
`--batch`/`--accum` control.

That ceiling is measured, not hoped for. `--dry-run` on this GPU, one
forward and one backward at r=16 on q_proj/v_proj:

    --batch  4      2.72 GB       --batch  4, 25-30 s clips   —
    --batch  8      3.74 GB       --batch  8, 25-30 s clips   3.82 GB
    --batch 12      4.83 GB
    --batch 16      5.88 GB       --batch 16, 25-30 s clips   7.05 GB

The right-hand column is why `--batch` defaults to 8. Cost per step is
not the mel — the feature extractor pads every clip to 30 s, so
`input_features` is always `(batch, 128, 3000)` — it is the decoder
logits, `(batch, labels, 51866)`, and `labels` is whatever the longest
transcript in the batch tokenises to. A batch of 16 long spontaneous
utterances reaches 7.05 GB of 7.73 usable, which fits only if nothing
else is on the card. At 8 the same batch costs 3.82 GB and leaves room
for the rest of the machine.

**Why the adapter is not the product.** `neiro` loads CTranslate2 from
`models/large-v3-turbo-ct2` through faster-whisper. A LoRA adapter — and
even a merged safetensors model — is invisible to it. The run ends by
printing the exact `ct2-transformers-converter` command, and until that
is run and `config.toml` points at the new directory, nothing the user
hears has changed.

**What the numbers mean.** Corpus-level WER from `neiro.evals.wer` —
total errors over total reference words, never the mean of per-utterance
rates — on the same held-out utterances before and after, so "did this
help" has an answer. Latency percentiles are nearest-rank p50/p95 via
`neiro.evals.latency.percentile`; there are no means in this file.
Kathbath's `valid` split is NOT speaker-disjoint from its train shards
(see `neiro.training.stt_data`), so its speakers are struck from the
training side and the count is printed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from neiro.evals.latency import percentile
from neiro.evals.wer import UtteranceScore, edit_distance, normalize, score
from neiro.training.licences import licences_for
from neiro.training.stt_data import (
    CORPORA,
    EVAL_SETS,
    SAMPLERATE,
    Row,
    SkipLog,
    SplitPlan,
    ct2_convert_command,
    decode_audio,
    index_corpus,
    pad_labels,
    plan_split,
    read_checkpoint_step,
    strip_decoder_start,
    total_hours,
    write_checkpoint_atomically,
)

MODEL_DIR = REPO / "models" / "large-v3-turbo"
CT2_DIR = REPO / "models" / "large-v3-turbo-ct2"
DATASETS_DIR = REPO / "data" / "datasets"
OUT_DIR = REPO / "models" / "stt-hindi-lora"
# Whisper's own limit: 448 decoder positions. A Hindi transcript never
# comes close, but a corrupt row could, and a label longer than the
# model's context is a silent truncation.
MAX_LABEL_TOKENS = 448


# --------------------------------------------------------------------------
# indexing


@dataclass
class Example:
    audio: np.ndarray
    text: str
    corpus: str
    name: str


class ParquetClips(IterableDataset):
    """Rows streamed out of parquet, one row group at a time.

    Never `load_dataset(...)` and never a list of decoded waveforms: 63
    GB does not fit anywhere on this machine. The index decides *which*
    rows belong to this partition; this walks the row groups those rows
    live in, decodes only them, and throws the buffer away.

    Shuffling is at row-group granularity plus within the group, not
    globally — a global shuffle would mean a random seek into a 500 MB
    parquet file per example. Row groups here hold 100 rows (Kathbath) to
    ~1,100 (IndicVoices), and their order is reshuffled every epoch from
    a seed that is saved in the report.
    """

    def __init__(
        self,
        rows: tuple[Row, ...],
        log: SkipLog,
        *,
        seed: int,
        shuffle: bool,
        read_batch: int,
    ) -> None:
        self.groups: dict[tuple[str, int], list[Row]] = {}
        for row in rows:
            self.groups.setdefault((row.shard, row.row_group), []).append(row)
        for group in self.groups.values():
            group.sort(key=lambda r: r.row_in_group)
        self.log = log
        self.seed = seed
        self.shuffle = shuffle
        self.read_batch = read_batch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle differently each epoch — and only from the seed, so
        a resumed or repeated run sees the same order."""
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(len(g) for g in self.groups.values())

    def __iter__(self):
        import pyarrow.parquet as pq

        keys = sorted(self.groups)
        if self.shuffle:
            import random

            random.Random(self.seed + self.epoch).shuffle(keys)

        info = torch.utils.data.get_worker_info()
        if info is not None:
            # Shard by row group, so two workers never open the same
            # group and every row is emitted exactly once.
            keys = keys[info.id :: info.num_workers]

        handles: dict[str, object] = {}
        for shard, group_index in keys:
            wanted = {r.row_in_group: r for r in self.groups[(shard, group_index)]}
            handle = handles.get(shard)
            if handle is None:
                try:
                    handle = handles[shard] = pq.ParquetFile(shard)
                except Exception as exc:  # noqa: BLE001
                    self.log.shard_failed(shard, f"{type(exc).__name__} at read time")
                    continue
            spec = CORPORA[self.groups[(shard, group_index)][0].corpus]
            offset = 0
            try:
                batches = handle.iter_batches(
                    batch_size=self.read_batch,
                    row_groups=[group_index],
                    columns=[spec.audio_column],
                )
                for batch in batches:
                    column = batch.column(spec.audio_column).to_pylist()
                    for i, cell in enumerate(column):
                        row = wanted.get(offset + i)
                        if row is None:
                            continue
                        blob = cell.get("bytes") if isinstance(cell, dict) else None
                        audio = decode_audio(blob, self.log)
                        if audio is None:
                            continue
                        yield Example(
                            audio=audio,
                            text=row.text,
                            corpus=row.corpus,
                            name=f"{row.corpus}-{Path(shard).stem}-{group_index}-{row.row_in_group}",
                        )
                    offset += len(column)
            except Exception as exc:  # noqa: BLE001 — a bad group is a note, not the end
                self.log.shard_failed(f"{shard}#rg{group_index}", f"{type(exc).__name__} mid-read")


class Collator:
    """Batch of waveforms + transcripts -> Whisper's inputs and labels.

    Three things that are easy to get wrong and produce no error:

    * the feature extractor pads (or truncates) every clip to 30 s, so
      the mel is always `(batch, 128, 3000)` regardless of clip length —
      this is why `--batch` costs the same whether the corpus is short
      read speech or long spontaneous speech;
    * the tokeniser must be told `language="hi", task="transcribe"`,
      otherwise the prefix tokens say English and the model is trained to
      contradict the prompt it will be given at runtime;
    * the leading `<|startoftranscript|>` is stripped, because the model
      prepends it itself — see `stt_data.strip_decoder_start`.
    """

    def __init__(self, processor, decoder_start_id: int) -> None:
        self.processor = processor
        self.decoder_start_id = decoder_start_id

    def __call__(self, batch: list[Example]) -> dict:
        features = self.processor.feature_extractor(
            [e.audio for e in batch],
            sampling_rate=SAMPLERATE,
            return_tensors="pt",
            # Marks which of the 3000 mel frames are real rather than the
            # extractor's padding. Training ignores it (the encoder is a
            # fixed 30 s window either way); `generate` warns without it,
            # and a warning that says results may be unreliable is not
            # something to leave in the log of a WER run.
            return_attention_mask=True,
        )
        encoded = self.processor.tokenizer([e.text for e in batch]).input_ids
        encoded = [ids[:MAX_LABEL_TOKENS] for ids in encoded]
        labels = pad_labels(strip_decoder_start(encoded, self.decoder_start_id))
        return {
            "input_features": features["input_features"],
            "attention_mask": features["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "texts": [e.text for e in batch],
            "corpora": [e.corpus for e in batch],
            "names": [e.name for e in batch],
        }


# --------------------------------------------------------------------------
# model


def build_model(args, device, dtype):
    """Load the trainable twin and wrap the attention projections in LoRA.

    `--lora-targets` names the projections because that choice is the
    experiment: q/v is the smallest thing that moves a Whisper fine-tune,
    and adding k/o roughly doubles the adapter for a change this GPU can
    afford to test but this recipe should not assume.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype)

    # Both are set for training, and neither is a formality. A forced
    # decoder prefix baked into the config fights the labels (which carry
    # their own prefix tokens); a non-empty suppress list is a generation
    # -time filter that has no meaning during teacher forcing and would
    # be carried into the saved config.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        # `--max-new-tokens` is the budget that matters, and the config's
        # `max_length` silently competes with it: transformers warns on
        # every single generate call and keeps only one of the two.
        model.generation_config.max_length = None

    if args.gradient_checkpointing:
        # `use_cache` and gradient checkpointing are mutually exclusive:
        # transformers warns and disables the cache anyway, but only
        # after the first forward pass has already allocated it.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Without this, every base parameter is frozen, the checkpointed
        # segment sees no input that requires grad, and backward raises
        # "element 0 of tensors does not require grad" — the standard
        # LoRA + checkpointing trap.
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_targets),
        bias="none",
    )
    model = get_peft_model(model, lora)

    # The adapter trains in fp32 on top of bf16 base weights. peft casts
    # the input to the adapter's dtype and the output back, so this costs
    # a few megabytes and removes the one place bf16 actually hurts:
    # AdamW's second moment over very small gradients.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    model.to(device)

    # Under LoRA every base parameter is already frozen, the convolution
    # stem included — asserted rather than re-frozen, because a
    # `freeze_feature_encoder()` call that silently did nothing would
    # read like a memory saving that was never made.
    encoder = model.base_model.model.model.encoder
    assert not encoder.conv1.weight.requires_grad, "conv stem is trainable — LoRA did not apply"
    assert not encoder.conv2.weight.requires_grad, "conv stem is trainable — LoRA did not apply"
    return model


def trainable_report(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / max(total, 1), 4),
    }


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


# --------------------------------------------------------------------------
# evaluation


@torch.no_grad()
def transcribe(
    model, loader, processor, device, args
) -> tuple[list[tuple[str, str, str]], list[str], list[float]]:
    """Greedy decode of every batch. Returns (name, reference, hypothesis)
    triples plus the corpus each came from and per-utterance latencies.
    """
    model.eval()
    was_cached = model.config.use_cache
    triples: list[tuple[str, str, str]] = []
    corpora: list[str] = []
    latencies: list[float] = []
    for batch in loader:
        features = batch["input_features"].to(device, dtype=model.dtype)
        started = time.perf_counter()
        ids = model.generate(
            input_features=features,
            attention_mask=batch["attention_mask"].to(device),
            language=args.language,
            task="transcribe",
            num_beams=args.beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        elapsed = (time.perf_counter() - started) * 1000 / max(len(batch["names"]), 1)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for name, reference, hypothesis, corpus in zip(
            batch["names"], batch["texts"], texts, batch["corpora"], strict=True
        ):
            triples.append((name, reference, hypothesis.strip()))
            corpora.append(corpus)
            latencies.append(elapsed)
    model.config.use_cache = was_cached
    return triples, corpora, latencies


def wer_table(triples, corpora, latencies) -> dict:
    """Corpus-level WER overall and per corpus, plus the worst utterances.

    `neiro.evals.wer.score` computes total errors over total reference
    words. Never the mean of per-utterance rates: a three-word command
    with one error scores 33% and would outweigh a thirty-word sentence
    with one error ten to one. Latency is nearest-rank p50/p95 for the
    same family of reason, and there is deliberately no mean anywhere.
    """
    overall = score(triples)
    per_corpus: dict[str, dict] = {}
    for name in sorted(set(corpora)):
        subset = [t for t, c in zip(triples, corpora, strict=True) if c == name]
        result = score(subset)
        per_corpus[name] = {
            "n": len(subset),
            "wer": None if math.isnan(result.wer) else round(result.wer, 4),
            "total_errors": result.total_errors,
            "total_ref_words": result.total_ref_words,
        }
    scored = sorted(
        (
            UtteranceScore(
                name=n,
                reference=r,
                hypothesis=h,
                errors=edit_distance(normalize(r), normalize(h)),
                ref_words=len(normalize(r)),
            )
            for n, r, h in triples
        ),
        key=lambda u: -(u.errors / max(1, u.ref_words)),
    )
    return {
        "n": len(triples),
        "wer": None if math.isnan(overall.wer) else round(overall.wer, 4),
        "total_errors": overall.total_errors,
        "total_ref_words": overall.total_ref_words,
        # Utterances the model answered with nothing: every reference
        # word a deletion. Two models at the same WER can differ entirely
        # here, and the difference is what a person hears.
        "rejected": sum(1 for _, _, h in triples if not h),
        "by_corpus": per_corpus,
        "stt_p50_ms": round(percentile(latencies, 0.50), 1),
        "stt_p95_ms": round(percentile(latencies, 0.95), 1),
        "worst": [
            {"name": u.name, "ref": u.reference[:90], "hyp": u.hypothesis[:90], "errors": u.errors}
            for u in scored[:5]
        ],
    }


def print_wer_table(title: str, before: dict, after: dict | None) -> None:
    print(f"\n{title}")
    print(
        f"  {'corpus':<16}{'n':>7}{'ref words':>11}{'WER before':>12}{'WER after':>11}{'delta':>9}"
    )
    rows = [("ALL", before, after)]
    for corpus in before.get("by_corpus", {}):
        rows.append(
            (corpus, before["by_corpus"][corpus], (after or {}).get("by_corpus", {}).get(corpus))
        )
    for name, b, a in rows:
        b_wer = b.get("wer")
        a_wer = (a or {}).get("wer")
        delta = "" if (b_wer is None or a_wer is None) else f"{(a_wer - b_wer) * 100:+8.2f}"
        print(
            f"  {name:<16}{b.get('n', 0):>7}{b.get('total_ref_words', 0):>11}"
            f"{'' if b_wer is None else f'{b_wer:>11.2%}'}"
            f"{'' if a_wer is None else f'{a_wer:>10.2%}'}{delta:>9}"
        )


# --------------------------------------------------------------------------
# arguments


def build_parser() -> argparse.ArgumentParser:
    """Every argument is saved into the checkpoint and the report, so a
    number in either can always be traced to the run that produced it.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpora", nargs="+", default=sorted(CORPORA), choices=sorted(CORPORA))
    ap.add_argument(
        "--eval-sets",
        nargs="+",
        default=list(EVAL_SETS),
        choices=list(EVAL_SETS),
        help="kathbath-valid is the corpus's own held-out shard; indicvoices-heldout "
        "is a speaker-hash slice, because IndicVoices ships train shards only",
    )
    ap.add_argument(
        "--eval-fraction",
        type=float,
        default=0.10,
        help="share of IndicVoices SPEAKERS held out, by sha256 bucket",
    )
    ap.add_argument(
        "--eval-limit",
        type=int,
        default=200,
        help="utterances scored per eval set; the whole held-out slice is 77 h, and "
        "a before/after number does not need it",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="cap the TRAINING set by audio hours, so a run can be scoped to an evening",
    )
    ap.add_argument(
        "--kathbath-share",
        type=float,
        default=None,
        help="fraction of --max-hours drawn from Kathbath (clean read speech); default "
        "is the natural mix, which is 86%% IndicVoices",
    )
    ap.add_argument("--min-seconds", type=float, default=0.3, help="shorter rows are dropped")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Whisper's window; a longer clip is truncated while its transcript is not",
    )
    ap.add_argument(
        "--keep-tagged",
        action="store_true",
        help="keep rows whose transcript carries annotator markup such as "
        "<unintelligible>; by default they are dropped and counted",
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=8,
        help="per-device batch; measured peak VRAM is 3.74 GB here and 7.05 GB at 16 "
        "with long transcripts, of 7.73 usable — see the module docstring",
    )
    ap.add_argument(
        "--accum",
        type=int,
        default=8,
        help="gradient accumulation steps; --batch x --accum is the effective batch",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optimiser steps; overrides --epochs when given",
    )
    ap.add_argument("--warmup", type=int, default=50, help="linear warmup steps")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-targets",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="attention projections LoRA wraps",
    )
    ap.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute activations instead of storing them; on 8 GB this is what "
        "makes --batch bigger than 1 possible",
    )
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--read-batch",
        type=int,
        default=64,
        help="rows pulled out of a parquet row group at a time; bounds the audio "
        "buffer, which is ~100 KB per row",
    )

    ap.add_argument("--language", default="hi")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--ct2-out", type=Path, default=None, help="default: <--out>/ct2")
    ap.add_argument("--quantization", default="float16", help="ct2 converter quantisation")
    ap.add_argument(
        "--adapter", type=Path, default=None, help="load an existing adapter before evaluating"
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=200,
        help=(
            "optimiser steps between checkpoint saves — a long run has no progress "
            "at all to resume from until the first one lands, so this is not "
            "optional the way it would be on a machine that never loses power"
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load <--out>/checkpoint if it exists and continue training from there",
    )
    ap.add_argument("--dry-run", action="store_true", help="index, build, one step, stop")
    ap.add_argument("--eval-only", action="store_true", help="the BEFORE number, with no training")
    ap.add_argument(
        "--publishable-only",
        action="store_true",
        help="refuse to train on any corpus whose weights_publishable is not yes",
    )
    return ap


# --------------------------------------------------------------------------
# main


def build_split(args, log: SkipLog) -> tuple[list[Row], SplitPlan]:
    rows: list[Row] = []
    for name in args.corpora:
        found = index_corpus(
            CORPORA[name],
            args.data_root,
            log,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            drop_tagged=not args.keep_tagged,
        )
        print(f"  {name}: {len(found)} rows, {total_hours(found):.1f} h")
        rows.extend(found)

    shares = None
    if args.kathbath_share is not None:
        shares = {"kathbath": args.kathbath_share, "indicvoices": 1.0 - args.kathbath_share}
        shares = {k: v for k, v in shares.items() if k in args.corpora}
    plan = plan_split(
        rows,
        log,
        eval_sets=[s for s in args.eval_sets if s.split("-")[0] in args.corpora],
        eval_fraction=args.eval_fraction,
        eval_limit=args.eval_limit,
        max_train_hours=args.max_hours,
        seed=args.seed,
        shares=shares,
    )
    return rows, plan


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    # What the weights this run produces may be used for, decided from
    # the corpora actually indexed rather than from what was asked for.
    licences = licences_for(sorted(args.corpora))
    restricted = [n for n, i in licences["per_corpus"].items() if i["weights_publishable"] != "yes"]
    if restricted and args.publishable_only:
        print(
            "Refusing to train: --publishable-only, and these corpora do not permit "
            f"publishable weights: {', '.join(restricted)}."
        )
        return 2

    log = SkipLog()
    print(f"indexing {', '.join(args.corpora)} under {args.data_root}")
    started = time.perf_counter()
    rows, plan = build_split(args, log)
    if not rows:
        print(
            f"No rows indexed under {args.data_root}. Run "
            "`neiro fetch-datasets --target hindi_stt` first, or pass --data-root.",
            file=sys.stderr,
        )
        for shard, reason in log.shards.items():
            print(f"  {shard}: {reason}", file=sys.stderr)
        return 2
    print(f"indexed {len(rows)} rows in {time.perf_counter() - started:.1f}s")
    print(json.dumps(plan.as_dict(), indent=1))
    if plan.leak_rows:
        print(
            f"NOTE: {plan.leak_rows} training rows dropped because their speaker is in an "
            f"eval set ({len(plan.leak_speakers)} speakers). Kathbath's `valid` split shares "
            "every one of its speakers with `train` — see neiro/training/stt_data.py."
        )
    if not plan.train.rows and not args.eval_only:
        print("Split left no training rows. Widen --max-hours or --corpora.", file=sys.stderr)
        return 2
    for partition in plan.evals.values():
        if not partition.rows:
            print(f"Eval set {partition.name} is empty — it would report NaN.", file=sys.stderr)
            return 2

    if not args.model.is_dir() or not any(args.model.glob("*.safetensors")):
        print(
            f"No trainable checkpoint at {args.model}.\n"
            "It is the safetensors twin of the ct2 runtime copy — fetch it with\n"
            "  uv run scripts/fetch_models.py --only large-v3-turbo\n"
            "and re-run. (The ct2 directory the daemon loads cannot be fine-tuned.)",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            f"torch {torch.__version__} cannot see a GPU. Fine-tuning 809M parameters on "
            "CPU would take weeks and the number would not be comparable. If this says "
            "'+cpu', the runtime project's CPU torch has leaked in through the editable "
            "path dependency — run `uv sync` in training/.",
            file=sys.stderr,
        )
        return 2
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"device: {device} ({torch.cuda.get_device_name()}), base dtype {dtype}")

    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        str(args.model), language=args.language, task="transcribe"
    )
    # Belt and braces: some processor versions ignore the constructor
    # kwargs, and a tokeniser that silently prefixes <|en|> trains the
    # model to contradict the prompt it gets at runtime.
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    model = build_model(args, device, dtype)
    if args.adapter is not None:
        model.load_adapter(
            str(args.adapter), adapter_name="default", is_trainable=not args.eval_only
        )
        print(f"loaded adapter {args.adapter}")
    params = trainable_report(model)
    print(
        f"parameters: {params['trainable'] / 1e6:.2f}M trainable of "
        f"{params['total'] / 1e6:.1f}M ({params['trainable_pct']}%)"
    )

    decoder_start = model.config.decoder_start_token_id
    collator = Collator(processor, decoder_start)

    def make_loader(rows: tuple[Row, ...], shuffle: bool, batch: int) -> DataLoader:
        clips = ParquetClips(rows, log, seed=args.seed, shuffle=shuffle, read_batch=args.read_batch)
        return DataLoader(
            clips,
            batch_size=batch,
            num_workers=args.workers,
            collate_fn=collator,
            drop_last=shuffle,
            persistent_workers=False,
        )

    # ---------------- dry run ----------------
    if args.dry_run:
        loader = make_loader(plan.train.rows or plan.eval_all.rows, shuffle=False, batch=args.batch)
        batch = next(iter(loader))
        features = batch["input_features"].to(device, dtype=dtype)
        labels = batch["labels"].to(device)
        print(
            f"batch: input_features {tuple(features.shape)} {features.dtype}, "
            f"labels {tuple(labels.shape)} (min {int(labels.min())}, max {int(labels.max())})"
        )
        model.train()
        started = time.perf_counter()
        out = model(input_features=features, labels=labels)
        forward_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        out.loss.backward()
        backward_ms = (time.perf_counter() - started) * 1000
        torch.cuda.synchronize()
        print(
            f"forward {forward_ms:.0f} ms -> logits {tuple(out.logits.shape)}, "
            f"loss {out.loss.detach().float():.4f}; backward {backward_ms:.0f} ms"
        )
        grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        print(f"{grads} adapter tensors received a gradient")
        print(
            f"peak VRAM: {peak_vram_gb()} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"rows skipped so far: {json.dumps(log.as_dict()['by_reason'])}")
        print(
            f"\nconvert the merged model with:\n  {shlex.join(ct2_convert_command(args.out / 'merged', args.ct2_out or args.out / 'ct2', args.quantization))}"
        )
        return 0

    # ---------------- before ----------------
    eval_loaders = {
        name: make_loader(partition.rows, shuffle=False, batch=args.batch)
        for name, partition in plan.evals.items()
    }
    before: dict[str, dict] = {}
    # The BEFORE number must be the BASE model. With the adapter freshly
    # initialised its B matrix is zero, so it is mathematically the base
    # model already — but that stops being true the moment --adapter loads
    # a trained one, and a baseline that silently included the thing being
    # measured is the worst kind of wrong.
    baseline = contextlib.nullcontext() if args.adapter else model.disable_adapter()
    with baseline:
        for name, loader in eval_loaders.items():
            triples, corpora, latencies = transcribe(model, loader, processor, device, args)
            before[name] = wer_table(triples, corpora, latencies)
            print_wer_table(f"BEFORE — {name}", before[name], None)

    if args.eval_only:
        _write_report(args, plan, log, licences, before, None, params, None)
        return 0

    # ---------------- train ----------------
    train_loader = make_loader(plan.train.rows, shuffle=True, batch=args.batch)
    per_epoch = max(1, len(plan.train.rows) // (args.batch * args.accum))
    total_steps = args.max_steps or per_epoch * args.epochs
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=min(0.3, args.warmup / max(total_steps, 1)),
    )
    print(f"training {total_steps} optimiser steps (batch {args.batch} x accum {args.accum})")

    checkpoint_dir = args.out / "checkpoint"
    resumed_step = 0
    if args.resume and checkpoint_dir.exists():
        # The checkpoint IS the adapter, already attached at this point (every
        # save below writes the live PeftModel, not a separate copy) — so
        # resuming is re-loading its weights onto the same wrapped model,
        # not building a second PeftModel on top of one that already has LoRA
        # layers. get_peft_model() further up already ran; this replaces its
        # (randomly initialised) adapter weights with the saved ones.
        model.load_adapter(str(checkpoint_dir), adapter_name="default", is_trainable=True)
        resumed_step = read_checkpoint_step(checkpoint_dir)
        print(
            f"resumed from {checkpoint_dir} at step {resumed_step} — optimiser and "
            "schedule restart fresh from here, which is a cheaper loss than the "
            "hours of compute a bare restart would throw away"
        )

    model.train()
    step = resumed_step
    running, seen = 0.0, 0
    started = time.perf_counter()
    peak = 0.0
    for epoch in range(args.epochs):
        train_loader.dataset.set_epoch(epoch)
        for micro, batch in enumerate(train_loader, 1):
            out = model(
                input_features=batch["input_features"].to(device, dtype=dtype),
                labels=batch["labels"].to(device),
            )
            (out.loss / args.accum).backward()
            running += float(out.loss.detach())
            seen += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip_grad
            )
            optimiser.step()
            schedule.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            peak = max(peak, peak_vram_gb() or 0.0)
            if step % 10 == 0 or step == 1:
                rate = step / max(time.perf_counter() - started, 1e-9)
                print(
                    f"  step {step}/{total_steps} loss {running / max(seen, 1):.4f} "
                    f"lr {schedule.get_last_lr()[0]:.2e} {rate * 60:.1f} steps/min "
                    f"peak {peak} GB"
                )
                running, seen = 0.0, 0
            if args.save_every and step % args.save_every == 0:
                # Overwrites in place rather than accumulating one directory
                # per save — a power cut has one checkpoint to lose, not
                # partial writes scattered across thirty of them filling the
                # disk. write_checkpoint_atomically saves to a temp dir and
                # renames over the live one, so a cut mid-save leaves the
                # PREVIOUS good checkpoint intact rather than a half-written
                # one that --resume would load broken.
                def _save(dest: Path) -> None:
                    model.save_pretrained(str(dest))
                    processor.save_pretrained(str(dest))

                write_checkpoint_atomically(_save, checkpoint_dir, step)
                print(f"  checkpoint saved at step {step} -> {checkpoint_dir}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    processor.save_pretrained(str(args.out / "adapter"))
    print(f"adapter saved to {args.out / 'adapter'}")

    # ---------------- after ----------------
    after: dict[str, dict] = {}
    for name, loader in eval_loaders.items():
        triples, corpora, latencies = transcribe(model, loader, processor, device, args)
        after[name] = wer_table(triples, corpora, latencies)
        print_wer_table(f"{name}", before[name], after[name])

    _write_report(args, plan, log, licences, before, after, params, peak)
    return 0


def _write_report(args, plan, log, licences, before, after, params, peak) -> None:
    merged = args.out / "merged"
    ct2_out = args.ct2_out or args.out / "ct2"
    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "split": plan.as_dict(),
        "skipped": log.as_dict(),
        "parameters": params,
        "peak_vram_gb": peak,
        "wer_before": before,
        "wer_after": after,
        # Travels with the weights, so an adapter found on disk a month
        # later still says what it may be used for.
        "licences": licences,
        "ct2_command": shlex.join(ct2_convert_command(merged, ct2_out, args.quantization)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nreport: {args.out / 'report.json'}")
    print(f"weights publishable: {licences['weights_publishable']}")
    print(f"rows skipped: {json.dumps(log.as_dict()['by_reason'])}")
    print(
        "\nThe adapter is not what the daemon loads. Merge and convert:\n"
        f'  python -c "from peft import PeftModel; from transformers import WhisperForConditionalGeneration as W; '
        f"m=PeftModel.from_pretrained(W.from_pretrained('{args.model}'), '{args.out / 'adapter'}').merge_and_unload(); "
        f"m.save_pretrained('{merged}')\"\n"
        f"  {shlex.join(ct2_convert_command(merged, ct2_out, args.quantization))}\n"
        f"then point cfg.stt.model_id at {ct2_out}.\n"
        "ct2-transformers-converter ships with `ctranslate2`, which is installed in the "
        f"RUNTIME venv ({REPO / '.venv'}), not this one."
    )


if __name__ == "__main__":
    sys.exit(main())
