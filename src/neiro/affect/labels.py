"""Mapping every emotion corpus's label set onto one arousal/valence plane.

**Why a plane and not a class list.** The thing Neiro needs from the
user's voice is `UserAffect(arousal_z, valence_z)` — how activated he
sounds and, weakly, whether it reads positive. It is emphatically *not*
one of her six VRM expressions: those are `NeiroState`, what *she*
feels, and CLAUDE.md rule 5 exists because merging the two is how an
assistant ends up mirroring its own face back at itself.

So a Lane B model trained here predicts two continuous values, and every
corpus's categorical labels are projected onto those two axes first.
That also makes corpora *combinable*: CREMA-D says "anger", ESD says
"angry", MELD says "anger", RAVDESS says "angry" with an intensity —
all of which are one point on the circumplex, and none of which agree on
a string.

**The coordinates.** Russell's circumplex, as used by essentially every
dimensional SER paper: valence on the horizontal, arousal on the
vertical, each in [-1, 1]. The numbers below are the standard consensus
positions, not tuned — tuning them to a corpus would bake that corpus's
acting style into the label definition.

**Acted is not natural.** `ACTED_CORPORA` is recorded because the
published gap is enormous: the best systems reach macro-F1 0.43 on
natural speech and 65-92 on acted contrast. A model validated only on
acted data will look excellent and then fail on a tired Tuesday. Any
evaluation that mixes them without saying so is lying.
"""

from __future__ import annotations

from dataclasses import dataclass


# valence, arousal — both in [-1, 1].
@dataclass(frozen=True)
class Circumplex:
    valence: float
    arousal: float


# The canonical positions. Neutral sits at the origin by definition;
# everything else is placed relative to it.
CIRCUMPLEX: dict[str, Circumplex] = {
    "neutral": Circumplex(0.0, 0.0),
    "calm": Circumplex(0.3, -0.5),
    "relaxed": Circumplex(0.4, -0.4),
    "bored": Circumplex(-0.3, -0.6),
    "tired": Circumplex(-0.2, -0.7),
    "sad": Circumplex(-0.6, -0.4),
    "depressed": Circumplex(-0.7, -0.5),
    "disgust": Circumplex(-0.6, 0.2),
    "contempt": Circumplex(-0.5, 0.1),
    "fear": Circumplex(-0.7, 0.7),
    "anger": Circumplex(-0.6, 0.8),
    "frustrated": Circumplex(-0.5, 0.4),
    "surprise": Circumplex(0.1, 0.8),
    "excited": Circumplex(0.6, 0.8),
    "happy": Circumplex(0.8, 0.5),
    "amused": Circumplex(0.7, 0.4),
}

# Every spelling every corpus in data/datasets.toml uses, normalised.
# Kept explicit rather than fuzzy-matched: a silent mismatch would train
# on wrong labels, and "ang" is not obviously "anger" to a regex.
ALIASES: dict[str, str] = {
    # CREMA-D uses three-letter codes in the filename.
    "ang": "anger",
    "dis": "disgust",
    "fea": "fear",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    # RAVDESS / TESS / SAVEE / ESD / EmoV-DB / MELD spellings.
    "angry": "anger",
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "afraid": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "surprise": "surprise",
    "ps": "surprise",
    "pleasant_surprise": "surprise",
    "pleasantsurprise": "surprise",
    "calm": "calm",
    "sleepiness": "tired",
    "sleepy": "tired",
    "amused": "amused",
    "amusement": "amused",
    "excitement": "excited",
    "exc": "excited",
    "fru": "frustrated",
    "frustration": "frustrated",
    "xxx": "",
    "other": "",
    "unknown": "",
    "none": "",
    # AI4Bharat Rasa's expressive styles are already canonical names
    # (anger / fear / joy / sad / neutral); "joy" is covered above.
}

# Corpora whose speech is performed, not spontaneous. The distinction is
# the single most important caveat in any SER number.
ACTED_CORPORA: frozenset[str] = frozenset(
    {"crema-d", "ravdess", "tess", "savee", "esd", "emov-db", "jl-corpus", "subesco", "rasa"}
)
NATURAL_CORPORA: frozenset[str] = frozenset({"msp-podcast", "iemocap", "meld", "msp-conversation"})


def normalise(label: str) -> str:
    """Corpus label → canonical name. Empty string when it maps to nothing.

    Empty rather than "neutral": an unusable label must be *dropped*, not
    silently trained as neutral. That mistake would teach the model that
    every ambiguous utterance is flat.
    """
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)
    return key if key in CIRCUMPLEX else ""


def to_circumplex(label: str) -> Circumplex | None:
    """Corpus label → (valence, arousal), or None if it maps to nothing."""
    name = normalise(label)
    return CIRCUMPLEX.get(name) if name else None


def is_acted(corpus: str) -> bool | None:
    """True acted, False natural, None unrecorded.

    `None` is deliberate and must stay visible: reporting a single
    accuracy over a mix of acted and natural speech without saying which
    is which is the most common way SER results mislead.
    """
    key = corpus.strip().lower()
    for known in ACTED_CORPORA:
        if key.startswith(known):
            return True
    for known in NATURAL_CORPORA:
        if key.startswith(known):
            return False
    return None
