"""Indexing Kathbath and IndicVoices for the Hindi fine-tune — everything
about the data that can be decided without torch.

`training/recipes/stt_train.py` needs CUDA torch and lives in the
training venv. The rules below decide what the model is trained on and
what a WER is computed over, and they are exactly the rules that are
worth a test, so they live here instead: importable by the runtime venv,
testable without a GPU, and shared by any later recogniser recipe.

**Which transcript column, and why.** Kathbath ships one (`text`).
IndicVoices ships three sanitized ones — `text`, `verbatim`,
`normalized` — plus two `unsanitized_*` variants. Measured on shard
`train-00003-of-00082` (5,429 rows): `text` and `normalized` are
**byte-identical in 100% of rows**, and `verbatim` differs from them in
43%. The difference is the speaker's own pronunciation: where a person
said "दर्सन", `verbatim` writes दर्सन and `normalized` writes the
dictionary spelling दर्शन; likewise मनुस्य / मनुष्य and ब्यवस्था /
व्यवस्था. The `unsanitized_*` pair is the same text with the annotator's
event markup left in — `<clinking>`, `<umm>`, `<inhaling>`,
`<Persistent-noise-start>` — which would teach the model to emit literal
angle-bracket tokens.

So: **`verbatim`**. A recogniser is asked what was said, not what should
have been said; training on `normalized` teaches it to silently correct
the speaker, which is a different task and one Elizabeth should not be doing
to a person talking to their own desktop. It is also what
`scripts/bench_stt.py`'s CORPORA table already scores IndicVoices
against, and a fine-tune whose training target disagrees with the
benchmark's reference is optimising something nobody measures.

**Kathbath's `valid` split is NOT speaker-held-out.** It looks like the
natural held-out set and it is not one: measured across all 34 shards,
**every one of its 20 speakers also appears in `train`**, covering
33,863 of the 91,752 train rows — 37%. (Kathbath comes from IndicSUPERB,
whose `valid` is the *known-speaker* validation set; the unknown-speaker
material is in test splits this mirror does not carry.) Fine-tuning on
all of `train` and reporting WER on `valid` would therefore report a
number for speakers the model had just been trained on, and it would
look like a clean held-out result. `plan_split` drops those speakers
from the training side and counts what it dropped; `SplitPlan` asserts
the two sides share no speaker.

**The index is (shard, row group, row), never audio.** Both corpora
together are 63 GB of flac inside parquet. Reading only
`speaker_id`/`duration`/transcript out of all 117 shards takes about a
second and ~200 MB, because parquet is columnar — so the whole corpus
can be indexed, split by speaker and capped by hours *before* a single
audio frame is decoded. The recipe then walks row groups and decodes
lazily.
"""

from __future__ import annotations

import io
import json
import random
import re
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from elizabeth.training.corpora import _bucket

SAMPLERATE = 16000

# The marker scripts/fetch_datasets.py leaves when a download finished.
# Its absence is not fatal — a partial corpus still yields rows — but a
# WER over whichever shards happened to arrive first deserves a line.
COMPLETE_MARKER = ".elizabeth-complete"

# Must match `_bucket`'s own default: the split is a hash bucket out of
# this many, and changing one without the other silently reshuffles
# which speakers are held out.
BUCKETS = 100

# What a masked label position is, for cross-entropy. Transformers'
# `shift_tokens_right` turns these back into the pad token when it builds
# `decoder_input_ids`, so the model never attends to a padded position
# and the loss never scores one.
IGNORE_INDEX = -100

# Whisper's feature extractor pads or truncates every clip to this many
# seconds. A longer clip is silently cut while its transcript keeps the
# words that were cut off — a label/audio mismatch the loss cannot see.
WHISPER_WINDOW_SECONDS = 30.0

# An annotator's markup, left in a sanitized transcript. The only one
# that survives IndicVoices' `verbatim` column is `<unintelligible>`
# (1,258 rows in the first five shards), which marks a stretch the human
# labeller could not make out: the audio contains speech the label does
# not. See `has_annotator_tag`.
_TAG = re.compile(r"<[^<>]{0,40}>")

# Every reason a row can be dropped. Naming them here rather than
# accepting any string means a typo is an error instead of a new bucket
# nobody reads — a silent one, because the count would still be printed.
SKIP_KINDS: tuple[str, ...] = (
    "missing_column",  # shard has no transcript/speaker/duration column
    "empty_text",  # transcript is blank
    "annotator_tag",  # transcript carries <unintelligible> or similar
    "too_long",  # would be truncated inside the 30 s window
    "too_short",  # below --min-seconds; usually a fragment or silence
    "no_duration",  # duration column present but null
    "no_audio",  # audio struct present but carries no bytes
    "decode_failed",  # soundfile could not read the bytes
    "speaker_leak",  # speaker also appears in an eval partition
)


@dataclass(frozen=True)
class CorpusSpec:
    """Where one corpus keeps the columns this recipe needs.

    Adding a corpus is an entry here, not a change to the indexing code.
    `text_column` is the deliberate choice (see the module docstring);
    `text_fallbacks` exist so a schema drift degrades to a different
    column visibly in the report rather than to an empty reference set,
    which is the same policy `scripts/bench_stt.py` uses.
    """

    name: str
    directory: str
    text_column: str
    audio_column: str
    speaker_column: str
    duration_column: str
    text_fallbacks: tuple[str, ...] = ()
    # Shard-filename prefixes this corpus publishes as a held-out split.
    # Empty means the corpus ships only training shards and any eval has
    # to be carved out by speaker.
    held_out_splits: tuple[str, ...] = ()


CORPORA: dict[str, CorpusSpec] = {
    "kathbath": CorpusSpec(
        name="kathbath",
        directory="kathbath/hindi",
        text_column="text",
        audio_column="audio_filepath",
        speaker_column="speaker_id",
        duration_column="duration",
        held_out_splits=("valid",),
    ),
    "indicvoices": CorpusSpec(
        name="indicvoices",
        directory="indicvoices/hindi",
        # See the module docstring. NOT `text`/`normalized` (identical to
        # each other, and both correct the speaker's pronunciation), and
        # NOT the `unsanitized_*` pair (annotator event markup).
        text_column="verbatim",
        audio_column="audio_filepath",
        speaker_column="speaker_id",
        duration_column="duration",
        text_fallbacks=("text", "normalized"),
    ),
}

# The two ways to be held out. `kathbath-valid` is the corpus's own
# shard (with its speakers removed from training, see the module
# docstring); `indicvoices-heldout` is a speaker-hash slice, because
# IndicVoices ships train shards only.
EVAL_SETS: tuple[str, ...] = ("kathbath-valid", "indicvoices-heldout")


@dataclass(frozen=True, slots=True)
class Row:
    """One utterance, addressed by where it sits in a parquet file.

    No audio: 63 GB of it stays on disk until a batch actually wants it.
    `seconds` comes from the corpus's own duration column, which is what
    makes hour capping and the 30 s guard possible before any decode.
    """

    corpus: str
    shard: str
    row_group: int
    row_in_group: int
    speaker: str  # corpus-namespaced, e.g. "kathbath:252"
    seconds: float
    text: str
    declared_split: str  # the shard filename's prefix: "train", "valid", ...

    @property
    def locator(self) -> tuple[str, int, int]:
        """Stable read order: shard, then row group, then row within it."""
        return (self.shard, self.row_group, self.row_in_group)


@dataclass
class SkipLog:
    """Rows and shards that could not be used, counted rather than fatal.

    A half-written shard, a null duration, a clip longer than the model's
    window — each is one row the recipe steps over. The count is the
    point: a run that silently dropped a third of a corpus and a clean
    run produce the same WER table, and only this tells them apart.
    """

    counts: Counter[str] = field(default_factory=Counter)
    shards: dict[str, str] = field(default_factory=dict)

    def skip(self, kind: str, n: int = 1) -> None:
        if kind not in SKIP_KINDS:
            raise ValueError(f"unknown skip kind {kind!r}; add it to SKIP_KINDS")
        self.counts[kind] += n

    def shard_failed(self, shard: str, reason: str) -> None:
        """A whole shard that could not be opened or has no usable columns."""
        self.shards[str(shard)] = reason

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {
            "total_rows_skipped": self.total,
            "by_reason": dict(sorted(self.counts.items())),
            "unusable_shards": dict(sorted(self.shards.items())),
        }


def speaker_key(corpus: str, speaker: object) -> str:
    """`corpus:speaker`, the same namespacing `corpora._emit` uses.

    Kathbath numbers its speakers from 38 and IndicVoices uses 16-digit
    strings, so a bare id would be ambiguous the moment a third corpus
    reuses an integer — and two different people merged into one speaker
    is exactly the leak the split exists to prevent.
    """
    return f"{corpus}:{speaker}"


def bucket_of(speaker: str) -> int:
    """Which of `BUCKETS` hash buckets a namespaced speaker falls in.

    Delegates to `corpora._bucket` — sha256, not Python's `hash()`, which
    is salted per process and would reshuffle the split on every run.
    """
    return _bucket(speaker, BUCKETS)


def shard_split(shard: str | Path) -> str:
    """The split a shard declares in its filename.

    `valid-00000-of-00002.parquet` -> "valid";
    `train-00007-of-00032.parquet` -> "train". An unrecognised name keeps
    whatever it says, so it can never be mistaken for a held-out split.
    """
    return Path(shard).stem.split("-", 1)[0]


def has_annotator_tag(text: str) -> bool:
    """Does this transcript carry `<unintelligible>` or similar markup?

    Such a row is a known label/audio mismatch: the annotator is saying
    there is speech here they could not transcribe. Trained on, it
    teaches the model to emit the literal word; scored against, it asks
    the model to guess a word the human could not hear.
    """
    return bool(_TAG.search(text or ""))


def text_for(record: Mapping[str, object], spec: CorpusSpec) -> str:
    """The transcript for one row, or "" when there is none.

    Tries the corpus's chosen column first, then its fallbacks, so a
    renamed column degrades to a different one — visibly, because the
    caller counts the rows that end up empty.
    """
    for column in (spec.text_column, *spec.text_fallbacks):
        value = record.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def index_shard(
    spec: CorpusSpec,
    shard: str | Path,
    row_groups: Sequence[Sequence[Mapping[str, object]]],
    log: SkipLog,
    *,
    min_seconds: float,
    max_seconds: float = WHISPER_WINDOW_SECONDS,
    drop_tagged: bool = True,
) -> list[Row]:
    """Turn one shard's cheap metadata columns into `Row`s.

    `row_groups[g]` is row group `g` as a list of dicts — whatever the
    parquet reader handed back. Keeping the pyarrow call outside means
    every rule below (which column, which rows are unusable, how a
    speaker is named) is testable in a venv with no pyarrow and no torch.
    """
    split = shard_split(shard)
    rows: list[Row] = []
    for group_index, group in enumerate(row_groups):
        for offset, record in enumerate(group):
            speaker = record.get(spec.speaker_column)
            duration = record.get(spec.duration_column)
            if speaker is None or spec.duration_column not in record:
                log.skip("missing_column")
                continue
            if duration is None:
                log.skip("no_duration")
                continue
            text = text_for(record, spec)
            if not text:
                log.skip("empty_text")
                continue
            if drop_tagged and has_annotator_tag(text):
                log.skip("annotator_tag")
                continue
            seconds = float(duration)
            if seconds > max_seconds:
                log.skip("too_long")
                continue
            if seconds < min_seconds:
                log.skip("too_short")
                continue
            rows.append(
                Row(
                    corpus=spec.name,
                    shard=str(shard),
                    row_group=group_index,
                    row_in_group=offset,
                    speaker=speaker_key(spec.name, speaker),
                    seconds=seconds,
                    text=text,
                    declared_split=split,
                )
            )
    return rows


# --------------------------------------------------------------------------
# reading the shards
#
# pyarrow, soundfile and librosa are imported inside the functions that
# need them: the runtime venv has the audio pair but no pyarrow, and the
# rules above have to stay importable there.


def select_shards(root: Path) -> list[Path]:
    """Every parquet shard under `root`, in a stable order.

    Hidden directories are excluded: HF's `.cache/` keeps lock and
    partial files next to real shards, and a `.parquet.incomplete` is
    exactly the file that must not be read.
    """
    return [
        p for p in sorted(root.rglob("*.parquet")) if not any(s.startswith(".") for s in p.parts)
    ]


def index_corpus(spec: CorpusSpec, data_root: Path, log: SkipLog, **limits) -> list[Row]:
    """Index one corpus without decoding a single audio frame.

    Parquet is columnar, so reading `speaker_id`, `duration` and the
    transcript out of all 117 shards of both corpora costs about a second
    and ~200 MB — while the audio those files also hold is 63 GB. That is
    the whole reason the split can be planned, asserted and capped before
    any training starts.
    """
    root = Path(data_root) / spec.directory
    if not root.is_dir():
        # Checked before pyarrow is imported, so "the corpus is not on
        # disk" is answered by the answer and not by an ImportError from
        # a venv that was never going to read parquet anyway.
        log.shard_failed(str(root), "not downloaded")
        return []

    import pyarrow.parquet as pq

    if not (root.parent / COMPLETE_MARKER).exists():
        print(
            f"  {root}: no {COMPLETE_MARKER} marker — indexing a partial download", file=sys.stderr
        )

    wanted = [spec.speaker_column, spec.duration_column, spec.text_column, *spec.text_fallbacks]
    rows: list[Row] = []
    for shard in select_shards(root):
        try:
            handle = pq.ParquetFile(shard)
        except Exception as exc:  # noqa: BLE001 — one bad shard is a note, not the end
            log.shard_failed(shard, f"{type(exc).__name__}: {exc}")
            continue
        present = [c for c in wanted if c in handle.schema_arrow.names]
        if spec.text_column not in present and not any(c in present for c in spec.text_fallbacks):
            log.shard_failed(shard, f"no transcript column in {handle.schema_arrow.names}")
            continue
        try:
            groups = [
                handle.read_row_group(g, columns=present).to_pylist()
                for g in range(handle.num_row_groups)
            ]
        except Exception as exc:  # noqa: BLE001
            log.shard_failed(shard, f"{type(exc).__name__} mid-read")
            continue
        rows.extend(index_shard(spec, shard, groups, log, **limits))
    return rows


# --------------------------------------------------------------------------
# streaming data


def decode_audio(blob: bytes | None, log: SkipLog) -> np.ndarray | None:
    """Encoded audio bytes -> mono float32 at 16 kHz, or None.

    The exact path every other module in this repo uses — soundfile from
    a BytesIO, mean down-mix, then `librosa.resample` — so a clip sounds
    the same to the trainer as it does to `scripts/bench_stt.py` and to
    the live provider. A row that cannot be read is counted and stepped
    over: a truncated flac somewhere in 63 GB must not end a run that has
    been going for an hour.
    """
    if not blob:
        log.skip("no_audio")
        return None
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 — one unreadable clip skips the row
        log.skip("decode_failed")
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLERATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLERATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def total_hours(rows: Iterable[Row]) -> float:
    return sum(r.seconds for r in rows) / 3600.0


def cap_by_hours(
    rows: Sequence[Row],
    max_hours: float | None,
    seed: int,
    shares: Mapping[str, float] | None = None,
) -> list[Row]:
    """Take rows until the audio-hour budget is spent.

    Shuffled first, deterministically: taking the first N rows in read
    order would take whole shards, and a shard is a handful of speakers
    recording in one session — so "six hours of Hindi" would mean "six
    hours of nine people".

    `shares` splits the budget between corpora by name (must sum to 1),
    for when the natural mix is not the wanted one: IndicVoices is 819 h
    of spontaneous, acoustically messy speech against Kathbath's 135 h of
    clean read speech, so an uncapped mix is 86% IndicVoices. `None`
    keeps the natural proportions.

    The budget is a ceiling on hours, not a row count, and the last row
    is admitted only if it fits — so a run scoped to an evening stays
    scoped to an evening.
    """
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    if max_hours is None:
        return sorted(ordered, key=lambda r: r.locator)

    if shares is not None:
        missing = {r.corpus for r in ordered} - set(shares)
        if missing:
            raise ValueError(f"shares must name every corpus present; missing {sorted(missing)}")
        if abs(sum(shares.values()) - 1.0) > 1e-6:
            raise ValueError(f"shares must sum to 1.0, not {sum(shares.values())}")
        budgets = {name: share * max_hours * 3600.0 for name, share in shares.items()}
    else:
        # One shared purse rather than one each: without `shares` the cap
        # is on the total and the corpora compete for it in shuffled
        # order, which reproduces their natural proportions.
        budgets = {}
    shared = max_hours * 3600.0

    kept: list[Row] = []
    spent: Counter[str] = Counter()
    total_spent = 0.0
    for row in ordered:
        if shares is not None:
            if spent[row.corpus] + row.seconds > budgets[row.corpus]:
                continue
        elif total_spent + row.seconds > shared:
            continue
        kept.append(row)
        spent[row.corpus] += row.seconds
        total_spent += row.seconds
    return sorted(kept, key=lambda r: r.locator)


@dataclass(frozen=True)
class Partition:
    """One named side of the split, and the counts that describe it."""

    name: str
    rows: tuple[Row, ...]

    @property
    def hours(self) -> float:
        return total_hours(self.rows)

    @property
    def speakers(self) -> frozenset[str]:
        return frozenset(r.speaker for r in self.rows)

    @property
    def by_corpus(self) -> dict[str, int]:
        counts: Counter[str] = Counter(r.corpus for r in self.rows)
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict:
        return {
            "n": len(self.rows),
            "hours": round(self.hours, 2),
            "speakers": len(self.speakers),
            "by_corpus": self.by_corpus,
        }


@dataclass(frozen=True)
class SplitPlan:
    """A speaker-independent partition of the indexed rows.

    Built by `plan_split`, which asserts the invariant rather than
    documenting it: no speaker appears on both the training side and any
    evaluation side.
    """

    train: Partition
    evals: dict[str, Partition]
    leak_rows: int
    leak_speakers: tuple[str, ...]
    train_hours_before_cap: float

    @property
    def eval_all(self) -> Partition:
        rows: list[Row] = []
        for partition in self.evals.values():
            rows.extend(partition.rows)
        return Partition("eval", tuple(sorted(rows, key=lambda r: r.locator)))

    def as_dict(self) -> dict:
        return {
            "train": self.train.as_dict(),
            "eval": {name: p.as_dict() for name, p in self.evals.items()},
            "train_hours_before_cap": round(self.train_hours_before_cap, 2),
            # Rows removed from training because their speaker is in an
            # eval partition. On Kathbath this is ~37% of the corpus and
            # the whole reason the number is printed.
            "speaker_leak_rows_dropped": self.leak_rows,
            "speaker_leak_speakers": len(self.leak_speakers),
        }


def plan_split(
    rows: Sequence[Row],
    log: SkipLog,
    *,
    eval_sets: Sequence[str] = EVAL_SETS,
    eval_fraction: float,
    eval_limit: int | None,
    max_train_hours: float | None,
    seed: int,
    shares: Mapping[str, float] | None = None,
) -> SplitPlan:
    """Partition indexed rows into training and one or more eval sets.

    Two kinds of held-out set, because the corpora are not the same
    shape. `kathbath-valid` is the shard Kathbath itself publishes —
    genuinely unseen *recordings*, but its speakers are also in `train`
    (see the module docstring), so every one of them is struck from the
    training side. `indicvoices-heldout` is a speaker-hash slice: buckets
    below `eval_fraction * BUCKETS`, using the same sha256 bucketing as
    `corpora.split`, so the same speaker lands on the same side on every
    machine and every run.

    The assertion at the end is the point of the function.
    """
    unknown = set(eval_sets) - set(EVAL_SETS)
    if unknown:
        raise ValueError(f"unknown eval set(s) {sorted(unknown)}; choose from {list(EVAL_SETS)}")
    if not eval_sets:
        raise ValueError("at least one eval set is needed; a fine-tune with no eval is a hope")

    edge = round(eval_fraction * BUCKETS)
    selected: dict[str, list[Row]] = {name: [] for name in eval_sets}
    eval_ids: set[tuple[str, int, int]] = set()
    for row in rows:
        if (
            "kathbath-valid" in selected
            and row.corpus == "kathbath"
            and row.declared_split in CORPORA["kathbath"].held_out_splits
        ):
            selected["kathbath-valid"].append(row)
            eval_ids.add(row.locator)
        elif (
            "indicvoices-heldout" in selected
            and row.corpus == "indicvoices"
            and bucket_of(row.speaker) < edge
        ):
            selected["indicvoices-heldout"].append(row)
            eval_ids.add(row.locator)

    # The held-out SPEAKERS are the split; `eval_limit` only decides how
    # many of their utterances are worth decoding. Taking the speaker set
    # from the full candidate pool rather than from the sampled rows is
    # what keeps the two independent — otherwise raising --eval-limit
    # after a run would quietly move speakers out of the training set the
    # adapter was already fitted on, and the "same held-out utterances"
    # the before/after table promises would not be the same data.
    eval_speakers: set[str] = {row.speaker for picked in selected.values() for row in picked}
    evals = {
        name: Partition(name, tuple(_take(picked, eval_limit, seed)))
        for name, picked in selected.items()
    }

    # Every row that did not land in an eval set is a training candidate
    # — including the rows of a held-out speaker whose own utterances
    # were dropped by `eval_limit`. Those are struck below by speaker,
    # not by row, which is the difference between a speaker-independent
    # split and one that merely looks like one.
    train_pool: list[Row] = []
    leak_speakers: set[str] = set()
    for row in rows:
        if row.locator in eval_ids:
            continue
        if row.speaker in eval_speakers:
            leak_speakers.add(row.speaker)
            log.skip("speaker_leak")
            continue
        train_pool.append(row)

    leak_rows = log.counts["speaker_leak"]
    before_cap = total_hours(train_pool)
    train = Partition("train", tuple(cap_by_hours(train_pool, max_train_hours, seed, shares)))

    overlap = train.speakers & eval_speakers
    assert not overlap, f"speaker leak: {sorted(overlap)[:5]}"
    return SplitPlan(
        train=train,
        evals=evals,
        leak_rows=leak_rows,
        leak_speakers=tuple(sorted(leak_speakers)),
        train_hours_before_cap=before_cap,
    )


def _take(rows: Sequence[Row], limit: int | None, seed: int) -> list[Row]:
    """At most `limit` rows, chosen deterministically, returned in read
    order.

    Shuffled before the cut so a limit does not mean "the first shard",
    then re-sorted by locator so the reader still walks each parquet file
    forwards.
    """
    ordered = list(rows)
    if limit is not None and len(ordered) > limit:
        random.Random(seed).shuffle(ordered)
        ordered = ordered[:limit]
    return sorted(ordered, key=lambda r: r.locator)


# --------------------------------------------------------------------------
# labels
#
# Expressed over python lists so they can be tested in a venv with no
# torch; the collator calls these and then wraps the result in a tensor,
# rather than re-deriving the same masking in torch where no test can
# reach it.


def strip_decoder_start(
    sequences: Sequence[Sequence[int]], decoder_start_id: int
) -> list[list[int]]:
    """Drop the leading `<|startoftranscript|>` the tokeniser adds.

    The model prepends it itself (`shift_tokens_right` builds
    `decoder_input_ids` from the labels), so leaving it in the labels
    trains the model to predict a token it is always given — the classic
    Whisper fine-tune off-by-one, which costs a little accuracy and
    produces no error at all.

    Only when **every** sequence starts with it. A batch where some do
    and some do not is a tokeniser surprise, and shifting half a batch by
    one position would be far worse than leaving it alone.
    """
    seqs = [list(s) for s in sequences]
    if seqs and all(s and s[0] == decoder_start_id for s in seqs):
        return [s[1:] for s in seqs]
    return seqs


def pad_labels(
    sequences: Sequence[Sequence[int]], ignore_index: int = IGNORE_INDEX
) -> list[list[int]]:
    """Pad a batch of label sequences to the longest, with `ignore_index`.

    Padding straight to -100 rather than to the pad token and masking
    afterwards: one step instead of two, and there is no intermediate
    state in which a pad token looks like a token to predict.
    """
    if not sequences:
        return []
    width = max(len(s) for s in sequences)
    return [list(s) + [ignore_index] * (width - len(s)) for s in sequences]


def ct2_convert_command(
    model_dir: str | Path, out_dir: str | Path, quantization: str = "float16"
) -> list[str]:
    """The command that turns the merged fine-tune back into what the
    daemon actually loads.

    A LoRA adapter, and even a merged safetensors model, is useless to
    the runtime: `elizabeth` loads CTranslate2 from `models/*-ct2` through
    faster-whisper. A fine-tune that is never converted is a report with
    no product. `ctranslate2` is installed in the RUNTIME venv, not the
    training one — this is a command to print, not to run from here.
    """
    return [
        "ct2-transformers-converter",
        "--model",
        str(model_dir),
        "--output_dir",
        str(out_dir),
        "--copy_files",
        "tokenizer.json",
        "preprocessor_config.json",
        "--quantization",
        quantization,
    ]


def write_checkpoint_atomically(
    save_fn: Callable[[Path], None], checkpoint_dir: Path, step: int
) -> None:
    """Save to a temp directory, then rename over the live checkpoint.

    A power cut during `save_fn` leaves `checkpoint_dir.tmp` half-written
    and `checkpoint_dir` itself untouched — `--resume` reads the old,
    complete checkpoint and loses at most `--save-every` steps, not the
    checkpoint it was about to lose anyway plus every step since the one
    before it. Writing straight into `checkpoint_dir` would risk exactly
    that: a cut mid-write corrupts the only copy `--resume` has to load.

    `os.rename` is atomic on both ends of this project's two OSes as long
    as source and destination share a filesystem — `checkpoint.tmp` and
    `checkpoint` are always siblings under the same `--out`, so that
    holds here without the caller having to think about it.
    """
    tmp_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    save_fn(tmp_dir)
    (tmp_dir / "step.json").write_text(json.dumps({"step": step}))
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def read_checkpoint_step(checkpoint_dir: Path) -> int:
    """The step a checkpoint was saved at, or 0 if it never recorded one.

    A checkpoint directory that exists but has no `step.json` (an older
    save, or one interrupted after the rename but before this file
    existed in an earlier version of the format) is not an error — it
    just means resuming re-plays from the start of the optimiser
    schedule, which is safe, only wasteful.
    """
    meta = checkpoint_dir / "step.json"
    if not meta.exists():
        return 0
    return int(json.loads(meta.read_text())["step"])
