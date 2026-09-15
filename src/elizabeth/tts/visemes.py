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

**The alphabet is misaki's, not the textbook's.** Kokoro's `pred_dur` is
one duration per *character* of the phoneme string, so `timeline()` has
to walk that string one character at a time — and misaki (Kokoro's G2P)
writes the English diphthongs and affricates as single letters precisely
so that one character is one sound: `A`=eɪ `I`=aɪ `O`=oʊ `W`=aʊ `Y`=ɔɪ
`Q`=əʊ `ʧ`=tʃ `ʤ`=dʒ `T`=ɾ. "hello" arrives as `həlˈO`, never as
`hɛloʊ`. A map keyed on the two-character spellings is dead on arrival:
every one of those letters used to fall through to the open-jaw default,
so "boy" and "go" opened the mouth wide where it should have rounded.

**Modifiers are not sounds.** Stress marks (`ˈ ˌ`), the length marks
(`ː ˑ`) and the combining diacritics describe the phone next to them.
Kokoro still tokenizes them and predicts a duration for each, so they
cannot be skipped — that duration belongs to a mouth shape. It is the
*previous* one: a length mark is by definition "keep holding what you
were saying", and a stress mark's slot is a few frames of transition.
Looking a modifier up as a phoneme is how `fˈuːd` used to flap the jaw
open in the middle of a rounded vowel.
"""

from __future__ import annotations

import unicodedata

VISEMES: tuple[str, ...] = ("aa", "ih", "ou", "ee", "oh")
SILENCE = ""

# What an unrecognised phoneme renders as. A shape, not silence: an
# unknown symbol still means the mouth is moving, and a mouth that
# freezes mid-word looks broken in a way a slightly wrong shape does not.
_FALLBACK = "aa"

# Everything misaki emits between words. `in` on a set, not on a string:
# `"!?" in " .,!?"` is a substring test and quietly says True for things
# that are not one symbol.
_PUNCTUATION = frozenset(" .,!?;:'\"-—…“”()")

# Symbols in Kokoro's token vocabulary that are not sounds and are not
# caught by their Unicode category: the tie bars are combining marks (Mn)
# and the length/stress marks are modifier letters (Lm), but the pitch
# arrows Kokoro uses for other languages are plain symbols.
_NOT_A_SOUND = frozenset("↓→↗↘")
# Mn = combining diacritic (nasal ̃, syllabic ̩, tie ͡), Lm = spacing
# modifier letter (ː ˑ ˈ ˌ ʰ ʲ), Sk = modifier symbol (rhotic hook ˞).
_MODIFIER_CATEGORIES = frozenset({"Mn", "Lm", "Sk"})

# IPA (as misaki/espeak emit it) -> VRM preset.
_MAP: dict[str, str] = {}


def _add(shape: str, phonemes: str) -> None:
    for p in phonemes.split():
        _MAP[p] = shape


# Open jaw, unrounded. `I W` are misaki's aɪ aʊ: both start open. `ᵊ` is
# its syllabic schwa ("bottle"), `ʔ` the glottal stop — the jaw stays
# where the surrounding vowel put it, which is open.
_add("aa", "ɑ a æ ʌ ɐ ɒ ɑː aɪ aʊ ɚ ɜ ɜː ə h I W ᵊ ʔ")
# Narrow, unrounded, front. `ʧ ʤ` are misaki's tʃ dʒ; `ɾ` and misaki's
# `T` are the flap in "butter", a tongue tap like t/d; `ᵻ` is the
# reduced vowel in "roses".
_add("ih", "ɪ i ɨ j s z t d n l r ɹ θ ð ʃ ʒ tʃ dʒ ʧ ʤ ʦ ʣ ʨ ʥ ɾ T ᵻ")
# Rounded, closed.
_add("ou", "u ʊ uː w f v")
# Wide, unrounded, front-high. `A` is misaki's eɪ.
_add("ee", "e ɛ eɪ ɛə A")
# Rounded, open. `O Y Q` are misaki's oʊ ɔɪ əʊ: all rounded.
_add("oh", "o ɔ oʊ ɔː ɔɪ m b p ŋ ɡ g k O Y Q")


def _is_modifier(symbol: str) -> bool:
    """True for a mark that shapes the phone beside it and is no phone itself.

    The explicit map wins: misaki's `ᵊ` is a modifier letter to Unicode
    but a real syllabic vowel to Kokoro, and it has a shape of its own.
    """
    if symbol in _MAP:
        return False
    return symbol in _NOT_A_SOUND or unicodedata.category(symbol) in _MODIFIER_CATEGORIES


def viseme_for(phoneme: str, previous: str = SILENCE) -> str:
    """The mouth shape for one phoneme. `""` for silence and punctuation.

    A bare modifier (`ː`, `ˈ`, a combining diacritic) has no shape of its
    own and returns `previous` — whatever the mouth was already doing.
    The default is silence: a modifier with nothing before it has nothing
    to extend, and the mouth had not opened yet.

    Unknown symbols fall back to `aa` rather than to silence: an
    unrecognised phoneme means the mouth should still be moving, and a
    mouth that freezes mid-word looks broken in a way a slightly wrong
    shape does not.
    """
    p = phoneme.strip()
    if not p or p in _PUNCTUATION:
        return SILENCE
    if p in _MAP:
        return _MAP[p]
    # A multi-character phoneme with diacritics or a length mark: the
    # base is what is left once the modifiers are gone.
    base = "".join(c for c in p if not _is_modifier(c))
    if not base:
        return previous
    return _MAP.get(base) or _MAP.get(base[0], _FALLBACK)


def timeline(
    phonemes: str, durations: list[float], offset: float = 0.0
) -> list[tuple[float, str, float]]:
    """`(start_seconds, viseme, duration_seconds)` per phoneme.

    Consecutive identical shapes are merged: "ss" is one long `ih`, not
    two, and re-triggering the same blendshape produces a visible
    stutter on the face.

    A modifier's duration extends the event before it. Kokoro predicts a
    duration for the `ː` in `uː` and for the `ˈ` in `fˈud` just as it
    does for a vowel, and the mouth should spend those frames holding
    the shape it is already in, not snapping to a default.
    """
    out: list[tuple[float, str, float]] = []
    t = offset
    for phoneme, duration in zip(phonemes, durations, strict=False):
        previous = out[-1][1] if out else SILENCE
        shape = viseme_for(phoneme, previous)
        if out and out[-1][1] == shape:
            start, _, held = out[-1]
            out[-1] = (start, shape, held + duration)
        else:
            out.append((t, shape, duration))
        t += duration
    return out
