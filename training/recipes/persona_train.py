#!/usr/bin/env python3
"""LoRA fine-tune Qwen3.5-4B on persona / Hinglish / tool-calling text --
the model this laptop already serves conversations with, picking up the
character prompt's voice from data instead of relying on the system
prompt alone.

    cd training
    uv run python recipes/persona_train.py --dry-run           # shapes, params, peak VRAM
    uv run python recipes/persona_train.py --max-hours 3.5

**Why QLoRA, not plain LoRA.** The base checkpoint's TEXT backbone alone
(`model.language_model.*` in the safetensors index; the vision tower and
the speculative-decoding MTP head are separate and this recipe never
loads either -- see below) is 4.206B parameters, 8.41 GB in bf16 --
already more than the ~7.73 GB usable on this 8 GB card, before a single
LoRA adapter, optimiser state or activation. Measured directly, not
assumed: summing every `model.language_model.*` tensor's shape x dtype
out of `model.safetensors.index.json` gives exactly that figure. A plain
bf16-base LoRA does not fit here; loading the frozen base in NF4 4-bit
(`bitsandbytes`, installed into this venv for exactly this reason -- see
DECISIONS below) brings it to ~2.3 GB and is what makes training
possible at all. This is QLoRA, not a stylistic choice.

**Which model class, and why it matters for VRAM.** The checkpoint's
`config.json` names `Qwen3_5ForConditionalGeneration` (a multimodal
class carrying a ~0.33B-parameter vision tower plus the 0.12B MTP head),
but `transformers` also registers a text-only `Qwen3_5ForCausalLM` for
the same `qwen3_5` config type, and `AutoModelForCausalLM` resolves to
it. Loading through `AutoModelForCausalLM` therefore never materialises
the vision tower or the MTP head at all -- their checkpoint keys are
simply left unloaded (`_keys_to_ignore_on_load_unexpected` on the class
already names `^mtp.*` and `^model.visual.*`) -- which is free VRAM this
recipe needs and none of our training data (text only) can use anyway.

**The architecture is a hybrid, and `q_proj`/`v_proj` alone would miss
three-quarters of it.** `config.json`'s `layer_types` is 32 entries,
24 `"linear_attention"` and 8 `"full_attention"` (every 4th layer).
Inspecting the loaded model's real module names (never guessed) shows
two entirely different attention blocks:

  * full-attention layers: `self_attn.{q,k,v,o}_proj` -- ordinary GQA;
  * linear-attention layers: `linear_attn.{in_proj_qkv,in_proj_a,
    in_proj_b,in_proj_z,out_proj}` -- a gated-delta-rule / linear
    attention block (`Qwen3_5GatedDeltaNet`), with `conv1d`, `A_log`,
    `dt_bias` and `norm` alongside them that are NOT `nn.Linear` and are
    deliberately not LoRA targets (a depthwise conv, two raw SSM
    parameter tensors, and an RMSNorm -- none of them a matrix LoRA's
    low-rank update composes with).

`--lora-targets` names the twelve real projection suffixes (attention,
both flavours, plus every layer's `mlp.{gate,up,down}_proj`); the
recipe turns that into a regex anchored to `model\\.layers\\.\\d+\\.` so
it can never accidentally match the (unloaded, but belt-and-braces) MTP
head's identically-named `self_attn.q_proj`. Measured on this model: 248
Linear4bit modules match (8 full-attn layers x 4 + 24 linear-attn layers
x 5 + 32 layers x 3 MLP projections), giving ~32.5M trainable LoRA
parameters at the default r=16 -- about 0.77% of the 4.206B-parameter
backbone.

**Thinking mode.** CLAUDE.md rule 4 says thinking is off and asserted
everywhere else in this project; this recipe asserts it for TRAINING
too, not just serving. Qwen3.5's `chat_template.jinja` IS a
hybrid-thinking template (it always renders the final assistant turn
inside a `<think>...</think>` scaffold) but with an important, verified
detail: none of our 437k prepared records carry a `reasoning_content`
field, and the template's fallback for that case is always an EMPTY
scaffold -- `<think>\\n\\n</think>\\n\\n` -- byte-identical to what the
server injects as a literal prompt prefix at serve time via
`chat_template_kwargs: {enable_thinking: false}`
(`neiro.llm.openai_compat.OpenAiCompatLlm.build_request`). So training
on this data trains the non-thinking path with no extra flag needed --
but this is *checked*, not assumed: `--dry-run` renders a real sample
and refuses to continue if a non-empty `<think>` block appears anywhere
in it, the same "assert, don't assume" policy rule 4 already applies at
runtime. Full reasoning in `neiro.training.llm_data`'s module docstring,
including why the standard incremental-prefix trick for chat-template
masking is UNSAFE on this specific template and what replaces it
(`{% generation %}` tags, transformers' own mechanism, verified
byte-identical to the vendor template's rendered text).

**Sequence length: measured, then clipped to what the card can hold.**
`--dry-run` and the real run both measure the token-length distribution
on a fresh `--length-sample`-record draw from `train.jsonl` (never
train on that measurement -- it is thrown away after choosing
`--max-length`) and report the `--length-percentile`-th nearest-rank
percentile, rounded up to `--length-round-to`. On the corpus this run
actually used: overall p50=643, p90=1316, p95=1410, p99=1606, max=2969
tokens (chat is longest -- p95=1469; emotion_text is short -- p95=49;
tool_call sits between -- p95=1075). The 0.90-percentile choice rounds
to 1344 -- but VRAM sweeps on THIS card, isolated per configuration to
avoid the caching allocator's own fragmentation muddying the reading,
show batch=1 stops being safe somewhere between 900 and 1024 tokens in
the worst realistic case (a short prompt + one long, almost entirely
un-masked assistant reply -- the regime where the fewest tokens are
masked away and the memory-efficient loss below buys the least). 1280
tokens at batch=1 held 6.90 GB even at a pessimistic 50%-masked
synthetic batch, with margin left for the run to not crash overnight on
one unlucky long example. So `--max-length-cap` defaults to 1280 and
wins over the measured percentile when the two disagree -- covering
~88% of records overall (79% chat, 98% tool_call, 100% emotion_text);
the rest are truncated from the FRONT (oldest turns dropped, the target
reply preserved) rather than skipped outright.

**The memory-efficient loss.** `model(input_ids=..., labels=...)`'s
built-in loss upcasts the FULL `(batch, seq_len, vocab_size)` logits to
float32 before computing cross-entropy -- and this model's vocabulary is
248,320 tokens, so at seq_len=1472 that one tensor alone is ~1.46 GB,
enough by itself to OOM a fresh 7.73 GB budget. `masked_loss` below
calls the backbone directly and applies the (unquantized, un-LoRA'd)
`lm_head` ONLY to the next-token-shifted positions the labels actually
supervise -- typically a small minority of the sequence, since most of
a training example is masked context. This changes nothing
mathematically (verified byte-identical against `model(...,
labels=...).loss` with LoRA dropout disabled) and is what makes a
max_length above a few hundred tokens fit on this card at all. An
earlier version of this function did not apply the standard next-token
shift and silently trained the model to predict its own current token
instead of the next one -- caught only by that byte-identical
comparison, not by any error.

**Behavioural eval, never pooled.** CLAUDE.md rule 8: this file never
averages tool-call accuracy and chat quality into one number, and
`by_kind` in `report.json` keeps `chat`, `tool_call` and `emotion_text`
as separate keys throughout, exactly the way `ser_train.py` treats
`by_speech_kind`. `chat` and `emotion_text` report a token-weighted
perplexity (total negative log-likelihood over total scored tokens,
never a mean of per-example rates -- the same aggregation the WER table
in `stt_train.py` uses). `tool_call` reports a structural well-formed
-call rate from real generation on held-out prompts, before and after,
on the same prompts -- `neiro.training.llm_data.check_tool_call_shape`.

**Licence.** This run calls `neiro.training.licences.licences_for` on
the sources actually present in `data/prepared/text/stats.json` (the
prep step's own record of what it read), not on an assumed list --
report it, don't guess it. `--publishable-only` drops only the
NON-publishable SOURCES' rows (train.jsonl interleaves many sources in
one file, unlike the STT recipe's `--corpora` which selects whole
corpora) and reports the count via `SkipLog`; without the flag, the run
proceeds and both the console and `report.json` say loudly whose
weights this makes the checkpoint not publishable for.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shlex
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from neiro.evals.latency import percentile
from neiro.training.licences import licences_for
from neiro.training.llm_data import (
    IGNORE_INDEX,
    SkipLog,
    build_training_chat_template,
    check_tool_call_shape,
    choose_max_length,
    filter_publishable,
    find_tool_call_turn,
    gguf_convert_commands,
    index_jsonl_lines,
    iter_jsonl,
    iter_jsonl_at_offsets,
    parse_tool_call_arguments,
    render_and_mask,
)

MODEL_DIR = REPO / "models" / "qwen3.5-4b-safetensors"
DATA_DIR = REPO / "data" / "prepared" / "text"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
STATS_PATH = DATA_DIR / "stats.json"
OUT_DIR = REPO / "models" / "persona-lora"

KINDS: tuple[str, ...] = ("chat", "tool_call", "emotion_text")

# The twelve real projection names -- see the module docstring for how
# these were found (never guessed): both attention flavours plus every
# layer's MLP. Anchored to `model.layers.N.` by the regex this builds,
# so it can never match the (unloaded) MTP head's identically-named
# self_attn.q_proj.
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_a",
    "in_proj_b",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# --------------------------------------------------------------------------
# data


class PersonaStream(IterableDataset):
    """Lazily reads, filters, renders and masks `train.jsonl` one line at
    a time -- 415,209 rows never sit tokenized in RAM together.

    Shuffling is a GLOBAL shuffle of a byte-offset index
    (`neiro.training.llm_data.index_jsonl_lines` /
    `iter_jsonl_at_offsets`), reseeded from `seed + epoch` on
    `set_epoch` (the same contract `stt_train.py`'s `ParquetClips.
    set_epoch` uses) -- not a streaming reservoir buffer. `train.jsonl`
    is one contiguous block per source (see `index_jsonl_lines`'s
    docstring), several of them bigger than any buffer this card could
    afford to hold as tokenized-adjacent state; a reservoir buffer fed
    nothing but GoEmotions for its first ~20k items can only ever emit
    GoEmotions. Indexing offsets once and shuffling THAT costs a few
    seconds and ~3.5 MB, and the read order it produces can start
    anywhere in the file.
    """

    def __init__(
        self,
        path: Path,
        tokenizer,
        chat_template: str,
        max_length: int,
        per_source: dict,
        log: SkipLog,
        *,
        publishable_only: bool,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.per_source = per_source
        self.log = log
        self.publishable_only = publishable_only
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._offsets: list[int] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _record_stream(self):
        if self.shuffle:
            if self._offsets is None:
                self._offsets = index_jsonl_lines(self.path)
            order = list(self._offsets)
            random.Random(self.seed + self.epoch).shuffle(order)
            return iter_jsonl_at_offsets(self.path, order)
        return iter_jsonl(self.path)

    def __iter__(self):
        records = self._record_stream()
        if self.publishable_only:
            records = filter_publishable(records, self.per_source, self.log)
        for record in records:
            messages = parse_tool_call_arguments(record["messages"], self.log)
            if messages is None:
                continue
            example = render_and_mask(
                self.tokenizer, messages, record.get("tools"), self.chat_template, self.max_length
            )
            if example is None:
                self.log.skip("no_assistant_tokens")
                continue
            yield example


class Collator:
    """Dynamic padding to the batch's own longest member -- `--batch`
    defaults to 1 (see the module docstring on why), so this mostly
    matters when a larger batch is asked for explicitly.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(example["input_ids"]) for example in batch)
        input_ids, labels, attention_mask = [], [], []
        for example in batch:
            pad = width - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * pad)
            labels.append(example["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append([1] * len(example["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def measure_lengths(path: Path, tokenizer, chat_template: str, sample: int, seed: int) -> list[int]:
    """Token lengths of `sample` records drawn from `path` via the same
    global offset shuffle training uses -- thrown away after
    `choose_max_length` reads it, never trained on.
    """
    log = SkipLog()
    offsets = index_jsonl_lines(path)
    order = list(offsets)
    random.Random(seed).shuffle(order)
    lengths: list[int] = []
    for record in iter_jsonl_at_offsets(path, order):
        if len(lengths) >= sample:
            break
        messages = parse_tool_call_arguments(record["messages"], log)
        if messages is None:
            continue
        # A template with {% generation %} tags makes apply_chat_template
        # return a BatchEncoding (dict-like: "input_ids", "attention_mask")
        # rather than a plain token list -- len() on THAT counts keys (2),
        # not tokens. Index by key, the same way render_and_mask does.
        rendered = tokenizer.apply_chat_template(
            messages, tools=record.get("tools") or None, chat_template=chat_template, tokenize=True
        )
        lengths.append(len(rendered["input_ids"]))
    return lengths


# --------------------------------------------------------------------------
# model


def build_model(args, device: torch.device):
    """Load the trainable model, quantized per the module docstring, and
    wrap the real attention+MLP projections in LoRA. Returns (model,
    tokenizer, chat_template, licences).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    chat_template = build_training_chat_template(tokenizer.chat_template)

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0} if device.type == "cuda" else None,
    )
    # The text-only class -- confirms the vision tower and MTP head were
    # never loaded. A future transformers release resolving `qwen3_5` to
    # the multimodal class instead would silently cost ~0.9 GB of VRAM
    # this recipe's budget does not have; this line is what would notice.
    print(f"model class: {type(model).__name__}")
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        print(
            "  WARNING: expected the text-only Qwen3_5ForCausalLM (see the module "
            "docstring) — the vision tower and/or MTP head may now be loaded too.",
            file=sys.stderr,
        )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    target_regex = (
        r"model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:"
        + "|".join(re.escape(name) for name in args.lora_targets)
        + r")$"
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, chat_template, target_regex


def masked_loss(model, input_ids, attention_mask, labels) -> tuple[torch.Tensor | None, int]:
    """See the module docstring's "memory-efficient loss" section. Returns
    (mean loss over scored tokens, number of scored tokens), or
    `(None, 0)` when nothing in the batch is scored at all.
    """
    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    hidden = backbone(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))
    flat_labels = shift_labels.reshape(-1)
    keep = flat_labels != IGNORE_INDEX
    n = int(keep.sum().item())
    if n == 0:
        return None, 0
    logits = lm_head(flat_hidden[keep])
    loss = F.cross_entropy(logits.float(), flat_labels[keep])
    return loss, n


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


# --------------------------------------------------------------------------
# evaluation — never pooled across kind, CLAUDE.md rule 8


def load_val_by_kind(path: Path, kinds: tuple[str, ...]) -> dict[str, list[dict]]:
    by_kind: dict[str, list[dict]] = {kind: [] for kind in kinds}
    for record in iter_jsonl(path):
        if record.get("kind") in by_kind:
            by_kind[record["kind"]].append(record)
    return by_kind


def sample_records(records: list[dict], limit: int, seed: int) -> list[dict]:
    import random

    if len(records) <= limit:
        return list(records)
    return random.Random(seed).sample(records, limit)


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, chat_template, records, max_length, device, log) -> dict:
    """Token-weighted perplexity over `records` (a single `kind`'s
    sample). Total negative log-likelihood over total scored tokens,
    never a mean of per-record rates — the same reason `stt_train.py`'s
    WER is `total_errors / total_ref_words`.
    """
    total_nll, total_tokens, scored = 0.0, 0, 0
    for record in records:
        messages = parse_tool_call_arguments(record["messages"], log)
        if messages is None:
            continue
        example = render_and_mask(
            tokenizer, messages, record.get("tools"), chat_template, max_length
        )
        if example is None:
            log.skip("no_assistant_tokens")
            continue
        ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
        loss, n = masked_loss(model, ids, None, labels)
        if loss is None:
            continue
        total_nll += float(loss) * n
        total_tokens += n
        scored += 1
    if total_tokens == 0:
        return {"n": scored, "scored_tokens": 0, "loss": None, "perplexity": None}
    mean_nll = total_nll / total_tokens
    return {
        "n": scored,
        "scored_tokens": total_tokens,
        "loss": round(mean_nll, 4),
        "perplexity": round(math.exp(min(mean_nll, 20.0)), 4),
    }


@torch.no_grad()
def evaluate_tool_calls(
    model, tokenizer, records, device, max_new_tokens, log, no_repeat_ngram_size: int
) -> dict:
    """Structural well-formed-call rate on real generations from held-out
    prompts — `neiro.training.llm_data.check_tool_call_shape`, never
    semantic equality (the task this checks does not need it).
    """
    n, well_formed, known = 0, 0, 0
    examples = []
    for record in records:
        turn = find_tool_call_turn(record["messages"])
        if turn is None:
            log.skip("no_assistant_tokens")
            continue
        prompt_messages = record["messages"][:turn]
        if not prompt_messages:
            continue
        tools = record.get("tools") or []
        declared = [t["function"]["name"] for t in tools if "function" in t]
        # return_dict=True: apply_chat_template returns a BatchEncoding
        # here too (a dict of "input_ids"/"attention_mask"), never a bare
        # tensor -- indexed by key, not assumed, the same lesson
        # measure_lengths above had to learn.
        rendered = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = rendered["input_ids"].to(device)
        attention_mask = rendered["attention_mask"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        check = check_tool_call_shape(text, declared or None)
        n += 1
        well_formed += int(check.well_formed)
        known += int(bool(check.function_known))
        if len(examples) < 3:
            examples.append({"generated": text[:200], "well_formed": check.well_formed})
    return {
        "n": n,
        "well_formed": well_formed,
        "well_formed_rate": round(well_formed / n, 4) if n else None,
        "known_function_rate": round(known / n, 4) if n else None,
        "examples": examples,
    }


# --------------------------------------------------------------------------
# arguments


def build_parser() -> argparse.ArgumentParser:
    """Every argument is saved into the checkpoint and the report, so a
    number in either can always be traced to the run that produced it.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", type=Path, default=TRAIN_PATH)
    ap.add_argument("--val", type=Path, default=VAL_PATH)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)

    ap.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="NF4 quantize the frozen base (bitsandbytes) — required: the bf16 text "
        "backbone alone is 8.41 GB, more than the ~7.73 GB usable on this card. See "
        "the module docstring.",
    )
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-targets",
        nargs="+",
        default=list(DEFAULT_LORA_TARGETS),
        help="the real attention+MLP projection names (both attention flavours); "
        "never guessed — see the module docstring for how these were found",
    )
    ap.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute activations instead of storing them; on 8 GB this is what "
        "makes a useful --max-length possible at all",
    )

    ap.add_argument("--batch", type=int, default=1, help="see the module docstring's VRAM sweep")
    ap.add_argument(
        "--accum",
        type=int,
        default=16,
        help="gradient accumulation; --batch x --accum is the effective batch",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--epochs", type=int, default=1, help="an upper bound — --max-hours will stop the run first"
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optimiser steps; stops the run if reached first",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=3.5,
        help="wall-clock cap. This laptop gets one evening, not an open-ended run — "
        "default is a weeknight's worth of GPU time, not a guess",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 keeps the offset-index shuffle simple",
    )
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--save-every", type=int, default=500, help="checkpoint the adapter every N optimiser steps"
    )

    ap.add_argument(
        "--max-length", type=int, default=None, help="override; skips the measurement below"
    )
    ap.add_argument(
        "--length-sample", type=int, default=3000, help="records measured to choose --max-length"
    )
    ap.add_argument("--length-percentile", type=float, default=0.90)
    ap.add_argument("--length-round-to", type=int, default=64)
    ap.add_argument(
        "--max-length-cap",
        type=int,
        default=1280,
        help="hard ceiling from the VRAM sweep in the module docstring; wins over the "
        "measured percentile when the two disagree",
    )

    ap.add_argument("--eval-limit", type=int, default=150, help="held-out records scored per kind")
    ap.add_argument(
        "--max-new-tokens", type=int, default=256, help="generation budget for the tool-call eval"
    )
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=0)

    ap.add_argument(
        "--publishable-only",
        action="store_true",
        help="drop rows from non-publishable SOURCES rather than refuse the whole run "
        "(train.jsonl interleaves many sources in one file — see the module docstring)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="load, build, one forward+backward, stop"
    )
    ap.add_argument(
        "--dry-run-min-batch",
        type=int,
        default=1,
        help="--dry-run halves --batch on OOM down to this floor before giving up",
    )
    ap.add_argument(
        "--gguf-quant", default="Q4_K_M", help="llama-quantize target type for the printed command"
    )
    return ap


# --------------------------------------------------------------------------
# main


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    if not args.model.exists():
        print(f"No trainable checkpoint at {args.model}.", file=sys.stderr)
        return 2
    if not args.train.exists() or not args.val.exists():
        print(
            f"No prepared data at {args.train} / {args.val}. Run "
            "`uv run python ../scripts/prep_text.py` from training/ first.",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            f"torch {torch.__version__} cannot see a GPU. This model does not fit in "
            "RAM-speed CPU training in any useful time, and the numbers would not be "
            "comparable anyway.",
            file=sys.stderr,
        )
        return 2
    device = torch.device("cuda")
    print(f"device: {device} ({torch.cuda.get_device_name()})")

    # What this run's weights may be used for — from the sources the prep
    # step actually read (data/prepared/text/stats.json), not an assumed
    # list; see the module docstring.
    sources: list[str] = []
    if args.stats.exists():
        stats = json.loads(args.stats.read_text())
        sources = sorted(stats.get("sources", {}))
    licences = licences_for(sources)
    restricted = [
        name
        for name, info in licences["per_corpus"].items()
        if info["weights_publishable"] != "yes"
    ]
    print(f"sources ({len(sources)}): {', '.join(sources)}")
    print(f"licence verdict: {licences['weights_publishable']}")
    for name in restricted:
        info = licences["per_corpus"][name]
        print(f"  restricted  {name}: {info['weights_publishable']} — {info['licence'][:80]}")
    if restricted:
        print(
            "NOTE: this checkpoint is NOT fully publishable — models/ is gitignored; "
            "the verdict travels in report.json regardless of --publishable-only."
        )

    model, tokenizer, chat_template, target_regex = build_model(args, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA target regex: {target_regex}")
    print(f"trainable LoRA parameters: {trainable / 1e6:.2f}M")

    # Assert the non-thinking training path, not assume it — CLAUDE.md
    # rule 4, applied to what the model is trained to DO.
    sample = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        chat_template=chat_template,
        tokenize=False,
    )
    if "<think>\n\n" not in sample and "<think>" in sample:
        print(
            f"REFUSING: a rendered sample contains a non-empty <think> block:\n{sample!r}\n"
            "Training would teach the model to reason mid-turn. See the module docstring.",
            file=sys.stderr,
        )
        return 2
    print(f"thinking-mode check ok — rendered sample: {sample!r}")

    max_length = args.max_length
    length_report = {"measured": False}
    if max_length is None:
        lengths = measure_lengths(
            args.train, tokenizer, chat_template, args.length_sample, args.seed
        )
        measured = choose_max_length(lengths, args.length_percentile, round_to=args.length_round_to)
        max_length = min(measured, args.max_length_cap)
        covered = sum(1 for n in lengths if n <= max_length) / len(lengths)
        length_report = {
            "measured": True,
            "sample_n": len(lengths),
            "percentile": args.length_percentile,
            "percentile_value": round(
                percentile([float(n) for n in lengths], args.length_percentile), 1
            ),
            "rounded_to": measured,
            "capped_to": max_length,
            "coverage_at_max_length": round(covered, 4),
        }
        print(
            f"max_length: measured p{args.length_percentile:.2f}={length_report['percentile_value']} "
            f"-> rounded {measured} -> capped {max_length} "
            f"(covers {covered * 100:.1f}% of the {len(lengths)}-record sample)"
        )
    else:
        print(f"max_length: {max_length} (explicit --max-length)")

    train_log = SkipLog()
    stream = PersonaStream(
        args.train,
        tokenizer,
        chat_template,
        max_length,
        licences["per_corpus"],
        train_log,
        publishable_only=args.publishable_only,
        shuffle=True,
        seed=args.seed,
    )
    collate = Collator(tokenizer.pad_token_id)

    if args.dry_run:
        batch = args.batch
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                loader = DataLoader(stream, batch_size=batch, num_workers=0, collate_fn=collate)
                example = next(iter(loader))
                ids = example["input_ids"].to(device)
                attn = example["attention_mask"].to(device)
                labels = example["labels"].to(device)
                print(
                    f"batch: input_ids {tuple(ids.shape)}, {int((labels != IGNORE_INDEX).sum())} scored tokens"
                )
                model.train()
                started = time.perf_counter()
                loss, n = masked_loss(model, ids, attn, labels)
                forward_ms = (time.perf_counter() - started) * 1000
                if loss is None:
                    print("this batch scored zero tokens — draw another dry-run sample")
                    return 2
                started = time.perf_counter()
                loss.backward()
                backward_ms = (time.perf_counter() - started) * 1000
                torch.cuda.synchronize()
                print(
                    f"forward {forward_ms:.0f} ms, loss {float(loss):.4f} over {n} tokens; backward {backward_ms:.0f} ms"
                )
                print(
                    f"peak VRAM: {peak_vram_gb()} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                )
                print(f"batch size that fit: {batch}")
                break
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if batch <= args.dry_run_min_batch:
                    print(
                        f"OOM even at batch={batch} (the floor). Lower --max-length and retry.",
                        file=sys.stderr,
                    )
                    return 2
                batch = max(args.dry_run_min_batch, batch // 2)
                print(f"OOM — retrying at batch={batch}")

        merged = args.out / "merged"
        convert, quantize = gguf_convert_commands(
            merged,
            args.out / "gguf-f16.gguf",
            args.out / f"gguf-{args.gguf_quant}.gguf",
            args.gguf_quant,
        )
        print("\nmerge + GGUF conversion, once trained (see the report's full commands too):")
        print(f"  {shlex.join(convert)}")
        print(f"  {shlex.join(quantize)}")
        return 0

    # ---------------- before ----------------
    val_by_kind = load_val_by_kind(args.val, KINDS)
    eval_log = SkipLog()
    before: dict[str, dict] = {}
    with model.disable_adapter():
        model.eval()
        for kind in ("chat", "emotion_text"):
            sample_rows = sample_records(val_by_kind[kind], args.eval_limit, args.eval_seed)
            before[kind] = evaluate_perplexity(
                model, tokenizer, chat_template, sample_rows, max_length, device, eval_log
            )
            print(f"BEFORE {kind}: {json.dumps(before[kind])}")
        sample_rows = sample_records(val_by_kind["tool_call"], args.eval_limit, args.eval_seed)
        before["tool_call"] = evaluate_tool_calls(
            model,
            tokenizer,
            sample_rows,
            device,
            args.max_new_tokens,
            eval_log,
            args.no_repeat_ngram_size,
        )
        print(f"BEFORE tool_call: well_formed_rate={before['tool_call']['well_formed_rate']}")

    # ---------------- train ----------------
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    total_steps = args.max_steps
    schedule = None
    if total_steps is not None:
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=args.lr, total_steps=total_steps
        )

    args.out.mkdir(parents=True, exist_ok=True)
    model.train()
    step, micro = 0, 0
    running_loss, running_tokens = 0.0, 0
    started = time.perf_counter()
    deadline = started + args.max_hours * 3600.0
    peak = 0.0
    stop_reason = "max_hours"
    for epoch in range(args.epochs):
        stream.set_epoch(epoch)
        for batch in loader:
            if time.perf_counter() >= deadline:
                stop_reason = "max_hours"
                break
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss, n = masked_loss(model, ids, attn, labels)
            if loss is None:
                continue
            (loss / args.accum).backward()
            running_loss += float(loss) * n
            running_tokens += n
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip_grad
            )
            optimiser.step()
            if schedule is not None:
                schedule.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            peak = max(peak, peak_vram_gb() or 0.0)
            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - started
                rate = step / max(elapsed, 1e-9)
                mean_loss = running_loss / max(running_tokens, 1)
                print(
                    f"  step {step}{f'/{total_steps}' if total_steps else ''} "
                    f"loss {mean_loss:.4f} {rate * 60:.1f} steps/min peak {peak} GB "
                    f"elapsed {elapsed / 3600:.2f}h"
                )
                running_loss, running_tokens = 0.0, 0
            if step % args.save_every == 0:
                model.save_pretrained(str(args.out / "adapter"))
                print(f"  checkpointed adapter at step {step}")
            if total_steps and step >= total_steps:
                stop_reason = "max_steps"
                break
        else:
            continue
        break

    model.save_pretrained(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    print(
        f"adapter saved to {args.out / 'adapter'} (stopped: {stop_reason}, {step} optimiser steps)"
    )

    # ---------------- after ----------------
    after: dict[str, dict] = {}
    model.eval()
    for kind in ("chat", "emotion_text"):
        sample_rows = sample_records(val_by_kind[kind], args.eval_limit, args.eval_seed)
        after[kind] = evaluate_perplexity(
            model, tokenizer, chat_template, sample_rows, max_length, device, eval_log
        )
        print(f"AFTER {kind}: {json.dumps(after[kind])}")
    sample_rows = sample_records(val_by_kind["tool_call"], args.eval_limit, args.eval_seed)
    after["tool_call"] = evaluate_tool_calls(
        model,
        tokenizer,
        sample_rows,
        device,
        args.max_new_tokens,
        eval_log,
        args.no_repeat_ngram_size,
    )
    print(f"AFTER tool_call: well_formed_rate={after['tool_call']['well_formed_rate']}")

    merged = args.out / "merged"
    f16_out = args.out / "gguf-f16.gguf"
    quant_out = args.out / f"gguf-{args.gguf_quant}.gguf"
    convert, quantize = gguf_convert_commands(merged, f16_out, quant_out, args.gguf_quant)
    merge_py = (
        f'python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM as M; '
        f"m=PeftModel.from_pretrained(M.from_pretrained('{args.model}', dtype='bfloat16'), "
        f"'{args.out / 'adapter'}').merge_and_unload(); m.save_pretrained('{merged}')\""
    )

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "max_length": max_length,
        "length_measurement": length_report,
        "sources": sources,
        "licences": licences,
        "trainable_parameters": trainable,
        "lora_target_regex": target_regex,
        "stop_reason": stop_reason,
        "optimiser_steps": step,
        "peak_vram_gb": peak,
        "before": before,
        "after": after,
        "train_skipped": train_log.as_dict(),
        "eval_skipped": eval_log.as_dict(),
        "merge_command": merge_py,
        "gguf_convert_command": shlex.join(convert),
        "gguf_quantize_command": shlex.join(quantize),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"\nreport: {args.out / 'report.json'}")
    print(f"weights publishable: {licences['weights_publishable']}")
    print("\nThe adapter is not what the daemon loads. Merge and convert:")
    print(f"  {merge_py}")
    print(f"  {shlex.join(convert)}")
    print(f"  {shlex.join(quantize)}")
    print(f"then point the LLM config at {quant_out}.")
    print(
        "convert_hf_to_gguf.py / llama-quantize are llama.cpp tools, not installed on "
        "this machine as of this run (checked `pacman -Qs llama` and `which "
        "convert_hf_to_gguf.py`) — install llama.cpp first, these commands are not run "
        "from here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
