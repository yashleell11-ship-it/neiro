"""IPA phonemes to VRM 1.0 mouth shapes.

VRM 1.0 has exactly five mouth presets — `aa ih ou ee oh` — and they are
**not** VRM 0.x's `A I U E O`. Getting the spelling wrong makes
`expressionManager.setValue()` silently do nothing, which is the single
most common way an avatar's mouth stays shut with no error anywhere.

Mapping ~40 English phonemes onto 5 shapes is lossy by construction. The
rule used here is the one animators use: group by **jaw openness and lip
rounding**, not by phonetic family. /f/ and /v/ are not their own shape
in VRM, so they go to the nearest narrow one rather than being dropped —
a mouth that stops moving mid-word reads as a bug, while a slightly
wrong shape does not.
"""

from __future__ import annotations

VISEMES: tuple[str, ...] = ("aa", "ih", "ou", "ee", "oh")
SILENCE = ""

# IPA (as misaki/espeak emit it) -> VRM preset.
_MAP: dict[str, str] = {}


def _add(shape: str, phonemes: str) -> None:
    for p in phonemes.split():
        _MAP[p] = shape


# Open jaw, unrounded.
_add("aa", "ɑ a æ ʌ ɐ ɒ ɑː aɪ aʊ ɚ ɜ ɜː ə h")
# Narrow, unrounded, front.
_add("ih", "ɪ i ɨ j s z t d n l r ɹ θ ð ʃ ʒ tʃ dʒ")
# Rounded, closed.
_add("ou", "u ʊ uː w f v")
# Wide, unrounded, front-high.
_add("ee", "e ɛ eɪ ɛə")
# Rounded, open.
_add("oh", "o ɔ oʊ ɔː ɔɪ m b p ŋ ɡ g k")


def viseme_for(phoneme: str) -> str:
    """The mouth shape for one phoneme. `""` for silence and punctuation.

    Unknown symbols fall back to `aa` rather than to silence: an
    unrecognised phoneme means the mouth should still be moving, and a
    mouth that freezes mid-word looks broken in a way a slightly wrong
    shape does not.
    """
    p = phoneme.strip()
    if not p or p in " .,!?;:'\"-—…":
        return SILENCE
    if p in _MAP:
        return _MAP[p]
    # Diacritics and length marks: try the base character.
    base = p[0]
    return _MAP.get(base, "aa")


def timeline(
    phonemes: str, durations: list[float], offset: float = 0.0
) -> list[tuple[float, str, float]]:
    """`(start_seconds, viseme, duration_seconds)` per phoneme.

    Consecutive identical shapes are merged: "ss" is one long `ih`, not
    two, and re-triggering the same blendshape produces a visible
    stutter on the face.
    """
    out: list[tuple[float, str, float]] = []
    t = offset
    for phoneme, duration in zip(phonemes, durations, strict=False):
        shape = viseme_for(phoneme)
        if out and out[-1][1] == shape:
            start, _, held = out[-1]
            out[-1] = (start, shape, held + duration)
        else:
            out.append((t, shape, duration))
        t += duration
    return out
