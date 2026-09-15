"""Turn downloaded emotion corpora into one table of labelled utterances.

Every corpus stores its labels somewhere different — CREMA-D in the
filename, RAVDESS in a numeric filename code, ESD in a per-speaker text
file, MELD in a CSV, EmoNet-Voice in a per-rater score cell inside a
parquet. This module hides that behind one row shape:

    Utterance(path, valence, arousal, corpus, speaker, label)

so `train.py` never learns a corpus's layout, and adding a corpus is one
parser function rather than a change to the training loop.

**Speaker-independent splits are not optional.** The same 91 actors in
CREMA-D each record the same 12 sentences in 6 emotions. Split those
rows at random and the model hears speaker 1001 saying "it's eleven
o'clock" angrily in train and sadly in test, and scores brilliantly by
recognising *the speaker and the sentence* rather than the emotion.
Published SER results have been inflated this way for years. `split()`
partitions by speaker, and a test asserts no speaker appears on both
sides.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from elizabeth.affect.labels import KINDS, Kind, is_acted, normalise, speech_kind, to_circumplex

DATASETS_DIR = Path(__file__).resolve().parents[3] / "data" / "datasets"


@dataclass(frozen=True)
class Utterance:
    path: str  # audio file, or "<archive>::<member>" for a file still inside a tarball
    valence: float
    arousal: float
    corpus: str
    speaker: str
    label: str  # CANONICAL name, never the corpus's own spelling
    raw_label: str = ""  # what the corpus actually said, kept for debugging

    @property
    def acted(self) -> bool | None:
        return is_acted(self.corpus)

    @property
    def kind(self) -> Kind | None:
        """acted / natural / synthetic — the bucket a score on this row
        belongs in. `acted` stays for the two-way question; this one
        exists because synthetic speech is neither answer to it.
        """
        return speech_kind(self.corpus)

    @property
    def in_archive(self) -> bool:
        return "::" in self.path


# --- per-corpus parsers -------------------------------------------------
#
# Each takes a corpus root and yields Utterances. They are deliberately
# tolerant: a corpus that is half-downloaded, or whose layout has changed,
# yields fewer rows rather than raising. The caller reports the count, and
# a count of zero is visible.

# CREMA-D: 1001_DFA_ANG_XX.wav = speaker _ sentence _ emotion _ intensity
_CREMA = re.compile(
    r"^(?P<speaker>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<level>[A-Z]{2})\.wav$"
)

# RAVDESS: 03-01-05-01-02-01-12.wav, field 3 is emotion, field 7 is actor
_RAVDESS = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(?P<actor>\d{2})\.wav$")
_RAVDESS_EMO = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

# TESS: OAF_back_angry.wav / YAF_dog_pleasant_surprise.wav
_TESS = re.compile(r"^(?P<speaker>OAF|YAF)_(?P<word>[a-z]+)_(?P<emo>[a-z_]+)\.wav$", re.IGNORECASE)

# SAVEE: DC_a01.wav — two-letter speaker, then an emotion letter + index.
_SAVEE = re.compile(r"^(?P<speaker>[A-Z]{2})_(?P<emo>[a-z]{1,2})\d+\.wav$")
_SAVEE_EMO = {
    "a": "anger",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprise",
}


def _emit(path: str, label: str, corpus: str, speaker: str) -> Utterance | None:
    """One row, with the label CANONICALISED.

    Storing each corpus's own spelling would make `summarise()`'s label
    counts meaningless the moment two corpora are mixed — which is the
    entire point of this module. CREMA-D's "ANG", RAVDESS's "05" and
    MELD's "anger" are one class, and the table has to say so.
    """
    point = to_circumplex(label)
    if point is None:
        return None  # unusable label — dropped, never called neutral
    return Utterance(
        path=path,
        valence=point.valence,
        arousal=point.arousal,
        corpus=corpus,
        speaker=f"{corpus}:{speaker}",  # namespaced: "1001" exists in more than one corpus
        label=normalise(label),
        raw_label=label,
    )


def _names_in(root: Path) -> list[tuple[str, str]]:
    """`(addressable_path, basename)` for every audio file under `root`,
    looking inside a single un-extracted tarball if that is all there is.

    HF ships several of these corpora as one `.tar.gz`. Indexing straight
    out of the archive means the index can be built (and inspected, and
    committed as counts) before spending disk and minutes on extraction.
    """
    out: list[tuple[str, str]] = []
    for pattern in ("*.wav", "*.flac", "*.mp3"):
        for p in root.rglob(pattern):
            out.append((str(p), p.name))
    if out:
        return out
    for archive in root.rglob("*.tar.gz"):
        try:
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getnames():
                    if member.lower().endswith((".wav", ".flac", ".mp3")):
                        out.append((f"{archive}::{member}", Path(member).name))
        except (tarfile.TarError, OSError):
            continue
    return out


def read_crema_d(root: Path) -> list[Utterance]:
    rows = []
    for path, name in _names_in(root):
        m = _CREMA.match(name)
        if m and (u := _emit(path, m["emo"], "crema-d", m["speaker"])):
            rows.append(u)
    return rows


def read_ravdess(root: Path) -> list[Utterance]:
    rows = []
    for path, name in _names_in(root):
        m = _RAVDESS.match(name)
        if not m:
            continue
        label = _RAVDESS_EMO.get(m.group(3), "")
        if label and (u := _emit(path, label, "ravdess", m["actor"])):
            rows.append(u)
    return rows


def read_tess(root: Path) -> list[Utterance]:
    rows = []
    for path, name in _names_in(root):
        m = _TESS.match(name)
        if m and (u := _emit(path, m["emo"], "tess", m["speaker"])):
            rows.append(u)
    return rows


def read_savee(root: Path) -> list[Utterance]:
    rows = []
    for path, name in _names_in(root):
        m = _SAVEE.match(name)
        if not m:
            continue
        label = _SAVEE_EMO.get(m["emo"], "")
        if label and (u := _emit(path, label, "savee", m["speaker"])):
            rows.append(u)
    return rows


# Rasa, after scripts/extract_rasa.py: <speaker>_<STYLE>_<n>.wav
_RASA = re.compile(r"^(?P<speaker>male|female)_(?P<emo>[A-Z]+)_\d+\.wav$")


def read_rasa(root: Path) -> list[Utterance]:
    """AI4Bharat Rasa — Hindi expressive speech, extracted to wavs.

    The only corpus here that is INDIAN, emotional, and CC-BY (so weights
    trained on it can ship). Its audio lives inside parquet shards, which
    this module cannot walk, so `scripts/extract_rasa.py` writes the six
    emotion styles out first; the other ten styles are reading registers
    (WIKI, BOOK, NEWS...) and are skipped rather than called neutral.

    Rasa ships one male and one female voice per language and no speaker
    ids, so gender stands in for speaker — which is exactly what the
    speaker-independent split needs from it.
    """
    rows = []
    for path, name in _names_in(root):
        m = _RASA.match(name)
        if m and (u := _emit(path, m["emo"], "rasa", m["speaker"])):
            rows.append(u)
    return rows


# EmoNet-Voice Bench, after scripts/extract_emonet.py: the wavs plus one
# `index.csv` beside them, a row per (clip, target category) carrying
# every expert's 0/1/2 score verbatim.
EMONET_CORPUS = "emonet-voice-bench"
EMONET_INDEX = "index.csv"
EMONET_INDEX_COLUMNS = ("file", "clip", "category", "intensities")


def parse_emonet_label(raw: str) -> tuple[str, tuple[int, ...]] | None:
    """EmoNet-Voice's `label` cell → (category, one intensity per rater).

    The cell is the Python repr of a list with one dict per expert:
    `[{'human-1': {'Shame': 1}}, {'human-4': {'Shame': 2}}]`, category
    URL-encoded ("Impatience%20and%20Irritability"). Every rater scores
    how strongly the clip's ONE target category is heard — 0 not at all,
    1 mildly, 2 intensely. `literal_eval`, never `eval`: the cell is
    bytes from a download.

    None when the cell does not parse, has no rater, or names more than
    one category. The caller drops such a row; guessing which category
    was meant is how a wrong label gets trained.
    """
    try:
        raters = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(raters, list) or not raters:
        return None
    categories: set[str] = set()
    intensities: list[int] = []
    for rater in raters:
        if not isinstance(rater, dict):
            return None
        for scores in rater.values():
            if not isinstance(scores, dict):
                return None
            for category, score in scores.items():
                if not isinstance(category, str) or isinstance(score, bool):
                    return None
                if not isinstance(score, int):
                    return None
                categories.add(unquote(category))
                intensities.append(score)
    if len(categories) != 1 or not intensities:
        return None
    return next(iter(categories)), tuple(intensities)


def emonet_agreed(intensities: tuple[int, ...]) -> bool:
    """Every rater heard the target category at all (score >= 1).

    This is the corpus's own inclusion rule, not a threshold chosen
    here: measured on the download, 4,692 of the 12,600 rows are
    unanimous, which is the bench size the EmoNet-Voice paper reports. A
    clip scored (2, 0) is a synthesis one expert heard and one did not;
    training it as its target category would teach the model that label
    from audio a listener could not find it in.
    """
    return bool(intensities) and all(score >= 1 for score in intensities)


def _emonet_intensities(cell: str) -> tuple[int, ...]:
    """The index's `intensities` cell, "1;2;2", → (1, 2, 2).

    Anything malformed → (), which `emonet_agreed` never accepts.
    """
    try:
        return tuple(int(part) for part in cell.split(";") if part != "")
    except ValueError:
        return ()


def read_emonet_voice_bench(root: Path) -> list[Utterance]:
    """EmoNet-Voice Bench — synthetic English, 42 fine categories scored
    by psychology experts, after `scripts/extract_emonet.py`.

    Why it is here: CC-BY, small, and its categories reach corners of
    the plane the acted sets never visit — contentment, distress,
    embarrassment, awe, disappointment. Why it is `synthetic`, not acted:
    every voice is generated, so a score on it says how well the model
    reads a TTS engine's idea of an emotion. `summarise()` counts it
    apart and `is_acted` answers None, so it can never be averaged into
    the acted or the natural number.

    **The speaker-independent split is weaker on this corpus.** It ships
    no voice id. The only identity in the data is the 8-hex prefix of
    the source filename, shared by the few segments of one generation,
    so that prefix is the pseudo-speaker — 11,447 of them across 12,600
    rows. The dozen or so generated voices therefore sit on both sides
    of any split, and a score here is partly voice familiarity. It
    cannot say what CREMA-D's 91-actor split says, and a test-set number
    on it must be read with that in mind.

    Labels: the extractor writes every rater's score; a row is used only
    when `emonet_agreed` — every expert heard the target category — and
    the category has a circumplex point. Rows whose wav is missing are
    skipped, so a half-written extraction yields fewer rows, not a crash
    in the middle of an epoch.
    """
    index = root / "extracted" / EMONET_INDEX
    if not index.is_file():
        return []
    rows = []
    with index.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            if not emonet_agreed(_emonet_intensities(record.get("intensities") or "")):
                continue
            clip = record.get("clip") or ""
            path = index.parent / (record.get("file") or "")
            if not clip or not path.is_file():
                continue
            if u := _emit(str(path), record.get("category") or "", EMONET_CORPUS, clip):
                rows.append(u)
    return rows


READERS = {
    "crema-d": read_crema_d,
    "ravdess": read_ravdess,
    "tess": read_tess,
    "savee": read_savee,
    "rasa": read_rasa,
    EMONET_CORPUS: read_emonet_voice_bench,
}


def load(names: list[str] | None = None, datasets_dir: Path | None = None) -> list[Utterance]:
    """Index every downloaded corpus this module knows how to read."""
    base = datasets_dir or DATASETS_DIR
    rows: list[Utterance] = []
    for name, reader in READERS.items():
        if names and name not in names:
            continue
        root = base / name
        if root.is_dir():
            rows.extend(reader(root))
    return rows


def ensure_extracted(name: str, datasets_dir: Path | None = None) -> Path:
    """Extract a corpus's archives once, into `<corpus>/extracted/`.

    Indexing can read member names straight out of a `.tar.gz`, which is
    what makes the table inspectable before committing disk. *Training*
    cannot: random access into a gzip stream is O(n) per read, so a
    training epoch over 7441 clips would decompress the whole archive
    7441 times.

    Idempotent — a marker file means a re-run is free.
    """
    base = datasets_dir or DATASETS_DIR
    root = base / name
    target = root / "extracted"
    marker = target / ".extracted"
    if marker.exists():
        return target
    archives = list(root.rglob("*.tar.gz")) + list(root.rglob("*.tgz"))
    if not archives:
        return root  # already plain files
    target.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tar:
            members = [
                m
                for m in tar.getmembers()
                # Refuse absolute paths and traversal: an archive is
                # untrusted input, and `..` in a member name writes
                # wherever it likes.
                if m.isfile() and not m.name.startswith(("/", "..")) and "/.." not in m.name
            ]
            tar.extractall(target, members=members, filter="data")
    marker.write_text(f"{len(archives)} archive(s)\n")
    return target


def _bucket(speaker: str, buckets: int = 100) -> int:
    """Stable hash of a speaker id. Python's `hash()` is salted per
    process, so using it would reshuffle the split on every run and leak
    test speakers into train across runs.
    """
    return int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % buckets


def split(
    rows: list[Utterance], val_fraction: float = 0.15, test_fraction: float = 0.15
) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
    """Partition BY SPEAKER into train/val/test.

    Not by row. See the module docstring: splitting CREMA-D's rows at
    random lets the model score by recognising the actor and the scripted
    sentence, which is how SER papers have inflated results for years.
    """
    val_edge = int(val_fraction * 100)
    test_edge = val_edge + int(test_fraction * 100)
    train, val, test = [], [], []
    for row in rows:
        bucket = _bucket(row.speaker)
        if bucket < val_edge:
            val.append(row)
        elif bucket < test_edge:
            test.append(row)
        else:
            train.append(row)
    return train, val, test


def summarise(rows: list[Utterance]) -> dict:
    """Counts that make a silent problem visible: zero rows, one corpus
    dominating, or acted, natural and synthetic speech being reported as
    if they were one thing.
    """
    by_corpus: dict[str, int] = {}
    by_label: dict[str, int] = {}
    by_kind: dict[str, int] = dict.fromkeys(KINDS, 0)
    for row in rows:
        by_corpus[row.corpus] = by_corpus.get(row.corpus, 0) + 1
        by_label[row.label] = by_label.get(row.label, 0) + 1
        if (kind := row.kind) is not None:
            by_kind[kind] += 1
    return {
        "n": len(rows),
        "speakers": len({r.speaker for r in rows}),
        "by_corpus": dict(sorted(by_corpus.items())),
        "by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
        **by_kind,
        "unrecorded": len(rows) - sum(by_kind.values()),
    }
