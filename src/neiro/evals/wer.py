"""Word error rate against Yash's own voice.

Why this exists (Gate G3a, Stage 0 Task 6): every accuracy number in the
research comes from datacentre benchmarks on other people's voices in
quieter rooms. Indian-accented English is out of domain for essentially
every model in this stack, and the Open ASR Leaderboard's own
Indian-accent column is likely cleaner audio than a hostel room with a
laptop mic. Thirty of your own utterances, transcribed by hand once, is
the only ground truth that will ever exist for this project — and every
STT swap for the life of the project gets scored against it.

No jiwer dependency: WER is a Levenshtein distance over word lists, and
writing it here means the normalisation rules are visible and arguable
rather than hidden in someone else's defaults.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# What survives normalisation, by Unicode general category: letters,
# combining marks, numbers — plus the apostrophe, because "don't" vs
# "dont" is a real difference in a transcript and collapsing it would
# flatter the model. Everything else is a word boundary.
#
# Categories rather than `\w`, and this is the whole reason: `\w` is
# `str.isalnum()`, and a Devanagari vowel sign is not alphanumeric. Under
# the old regex "मेरे" came out as "म र" and "क़िला" as "क ल" — every
# matra and every nukta stripped, every Hindi word broken into consonant
# fragments, and a perfect Hindi transcript did not score zero. Nothing
# in English ever exercised that path, so nothing noticed.
_KEEP = frozenset("LMN")  # first letter of the general category
# ZWJ / ZWNJ shape how a conjunct renders; they are not sounds. Removed
# rather than spaced, so "रेल‌वे" stays one word instead of becoming two
# errors.
_DROP = "Cf"
_APOSTROPHE = "'"


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, split to words.

    Deliberately simple and deliberately visible. Two things it does NOT
    do, both of which would quietly lower the reported WER: expand
    numerals ("20" vs "twenty" stays an error, and so does "२०" vs "बीस")
    and expand contractions. A number that flatters the model is worse
    than no number.

    A third thing it does not do, for Hindi: **transliterate**. A
    reference in Devanagari against a hypothesis in Latin script scores
    as every word wrong, and that is the correct score — the recogniser
    was asked what was said and answered in the wrong script, which is
    precisely the failure a Hindi speaker hits. Mapping "मैं" onto "main"
    to forgive it would report a Hindi WER that nobody experiences.
    Hinglish written in Latin script is scored as the Latin words it is.

    NFC, not NFKC: canonical composition makes the two spellings of a
    nukta letter (precomposed U+0958, or क followed by U+093C) compare
    equal, and puts a nukta and a virama into canonical order when an
    input method stacked them the other way round — encoding, not an
    error. Compatibility folding buys nothing for a transcript.
    Punctuation goes by category, so the danda (।), the double danda
    (॥) and the abbreviation sign (॰) are stripped exactly as ASCII
    punctuation is, while every matra, nukta, anusvara and candrabindu
    stays attached to its word.
    """
    text = unicodedata.normalize("NFC", text).lower()
    kept: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in _KEEP or ch == _APOSTROPHE:
            kept.append(ch)
        elif category != _DROP:
            kept.append(" ")
    return "".join(kept).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over word lists (substitutions, insertions,
    deletions all cost 1).
    """
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    # single-row DP, O(len(hyp)) memory
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate for one utterance. 0.0 is perfect.

    Can exceed 1.0 when the hypothesis is longer than the reference and
    wrong — that's correct, not a bug, and it's why the corpus-level
    aggregate below is computed from raw error counts rather than by
    averaging these.
    """
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        # No reference words: any output at all is pure insertion error.
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


@dataclass
class UtteranceScore:
    name: str
    reference: str
    hypothesis: str
    errors: int
    ref_words: int

    @property
    def wer(self) -> float:
        return self.errors / self.ref_words if self.ref_words else 0.0


@dataclass
class DatasetScore:
    utterances: list[UtteranceScore]

    @property
    def total_errors(self) -> int:
        return sum(u.errors for u in self.utterances)

    @property
    def total_ref_words(self) -> int:
        return sum(u.ref_words for u in self.utterances)

    @property
    def wer(self) -> float:
        """Corpus-level WER: total errors over total reference words.

        NOT the mean of per-utterance WERs — that would weight a
        three-word command the same as a thirty-word ramble, and is the
        single most common way a reported WER ends up wrong.

        An empty corpus is **NaN, not 0.0**. Zero errors over zero words
        is not a perfect score, it is no measurement — and 0.0 would
        print as "WER 0.0%", which is indistinguishable from a flawless
        run and is exactly what a silently-empty dataset produces.
        """
        if not self.total_ref_words:
            return float("nan")
        return self.total_errors / self.total_ref_words


def score(pairs: list[tuple[str, str, str]]) -> DatasetScore:
    """Score a list of (name, reference, hypothesis) triples."""
    scored = []
    for name, reference, hypothesis in pairs:
        ref_words = normalize(reference)
        errors = edit_distance(ref_words, normalize(hypothesis))
        scored.append(
            UtteranceScore(
                name=name,
                reference=reference,
                hypothesis=hypothesis,
                errors=errors,
                ref_words=len(ref_words),
            )
        )
    return DatasetScore(utterances=scored)


def load_references(refs_path: Path) -> list[dict]:
    """Read a refs.jsonl written by `neiro record-set`."""
    entries = []
    with refs_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
