"""Turn downloaded emotion corpora into one table of labelled utterances.

Every corpus stores its labels somewhere different — CREMA-D in the
filename, RAVDESS in a numeric filename code, ESD in a per-speaker text
file, MELD in a CSV. This module hides that behind one row shape:

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

import hashlib
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

from neiro.affect.labels import is_acted, normalise, to_circumplex

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


READERS = {
    "crema-d": read_crema_d,
    "ravdess": read_ravdess,
    "tess": read_tess,
    "savee": read_savee,
    "rasa": read_rasa,
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
    dominating, or acted data being reported as if it were natural.
    """
    by_corpus: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for row in rows:
        by_corpus[row.corpus] = by_corpus.get(row.corpus, 0) + 1
        by_label[row.label] = by_label.get(row.label, 0) + 1
    acted = sum(1 for r in rows if r.acted is True)
    natural = sum(1 for r in rows if r.acted is False)
    return {
        "n": len(rows),
        "speakers": len({r.speaker for r in rows}),
        "by_corpus": dict(sorted(by_corpus.items())),
        "by_label": dict(sorted(by_label.items(), key=lambda kv: -kv[1])),
        "acted": acted,
        "natural": natural,
        "unrecorded": len(rows) - acted - natural,
    }
