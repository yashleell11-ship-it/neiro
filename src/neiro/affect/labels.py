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
evaluation that mixes them without saying so is lying. Synthetic (TTS)
speech is a third kind, not a flavour of acted: `SYNTHETIC_CORPORA` and
`speech_kind()` keep it out of both buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    # EmoNet-Voice (Schuhmann et al. 2025) rates 42 fine categories, and
    # these are the ones with no point above. Placed the same way: the
    # quadrant and rough distance the dimensional literature gives the
    # word (Russell 1980; Scherer's Geneva Emotion Wheel), not fitted to
    # the corpus. Two names on one point is fine — the model regresses
    # the point, and the name survives in `Utterance.raw_label`.
    "affection": Circumplex(0.7, 0.0),
    "grateful": Circumplex(0.6, 0.1),
    "content": Circumplex(0.6, -0.3),
    "pleasure": Circumplex(0.7, 0.2),
    "hopeful": Circumplex(0.5, 0.2),
    "interested": Circumplex(0.4, 0.3),
    "awe": Circumplex(0.4, 0.5),
    "confused": Circumplex(-0.2, 0.3),
    "doubtful": Circumplex(-0.3, 0.1),
    "longing": Circumplex(-0.3, -0.2),
    "disappointed": Circumplex(-0.5, -0.2),
    "ashamed": Circumplex(-0.5, -0.1),
    "embarrassed": Circumplex(-0.4, 0.3),
    "distressed": Circumplex(-0.7, 0.6),
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
    #
    # EmoNet-Voice (t1a5anu-anon/emonet-voice-bench) spells its 42
    # categories as capitalised words, URL-encoded inside the parquet
    # ("Impatience%20and%20Irritability"); `corpora.parse_emonet_label`
    # decodes them before they get here. Each lands on the nearest point
    # above — a name that already has one (Anger, Disgust, Fear, Sadness,
    # Amusement, Contempt, Awe, Longing, Pleasure) needs no entry.
    "astonishment": "surprise",
    "bitterness": "contempt",
    "confusion": "confused",
    "contentment": "content",
    "disappointment": "disappointed",
    "distress": "distressed",
    "doubt": "doubtful",
    "elation": "excited",
    "embarrassment": "embarrassed",
    "fatigue": "tired",
    "helplessness": "sad",
    "hope": "hopeful",
    "impatience_and_irritability": "frustrated",
    "infatuation": "happy",
    "interest": "interested",
    "jealousy_&_envy": "frustrated",
    "malevolence": "contempt",
    "pain": "distressed",
    "pride": "happy",
    "relief": "content",
    "shame": "ashamed",
    "sourness": "disgust",
    "teasing": "amused",
    "thankfulness": "grateful",
    "triumph": "excited",
    # The rest of EmoNet's list are not emotional states: two perceptual
    # dimensions (one of them literally named after the arousal axis),
    # two cognitive states, a drive, an intoxicant and the absence of
    # affect. None has a defensible point on the plane, so each is
    # dropped — the same rule as "xxx" above, never "neutral".
    "arousal": "",
    "authenticity": "",
    "concentration": "",
    "contemplation": "",
    "emotional_numbness": "",
    "intoxication": "",
    "sexual_lust": "",
}

# Corpora whose speech is performed, not spontaneous. The distinction is
# the single most important caveat in any SER number.
ACTED_CORPORA: frozenset[str] = frozenset(
    {"crema-d", "ravdess", "tess", "savee", "esd", "emov-db", "jl-corpus", "subesco", "rasa"}
)
NATURAL_CORPORA: frozenset[str] = frozenset({"msp-podcast", "iemocap", "meld", "msp-conversation"})
# Generated by a TTS engine — neither performed by a person nor
# spontaneous. A number on synthetic speech says how well the model reads
# an engine's idea of an emotion, which is a third question, not a
# harder or easier version of the other two.
SYNTHETIC_CORPORA: frozenset[str] = frozenset({"emonet-voice-bench"})

Kind = Literal["acted", "natural", "synthetic"]
KINDS: tuple[Kind, ...] = ("acted", "natural", "synthetic")


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


def speech_kind(corpus: str) -> Kind | None:
    """acted / natural / synthetic, or None when the corpus is unrecorded.

    Prefix-matched because manifest slugs carry versions
    ("msp-podcast-v2-0"). `None` is deliberate and must stay visible:
    reporting one accuracy over a mix of kinds without saying which is
    which is the most common way SER results mislead.
    """
    key = corpus.strip().lower()
    for name, members in (
        ("acted", ACTED_CORPORA),
        ("natural", NATURAL_CORPORA),
        ("synthetic", SYNTHETIC_CORPORA),
    ):
        if any(key.startswith(known) for known in members):
            return name
    return None


def is_acted(corpus: str) -> bool | None:
    """True acted, False natural, None otherwise.

    Synthetic speech answers None too: it is neither, and a bool here
    would let it be averaged into one of the two buckets the recipe
    reports apart. Ask `speech_kind` when the third answer matters.
    """
    kind = speech_kind(corpus)
    if kind == "acted":
        return True
    if kind == "natural":
        return False
    return None
