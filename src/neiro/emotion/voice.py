"""Her emotional state reaching her voice.

The other consumer of the same `NeiroState` that drives the face. If the
face and the voice disagree — a bright expression over a flat reading —
the effect is worse than having neither, because the mismatch is exactly
what people notice.

**Why a free-form string.** The verification pass killed the spec's TTS
plan: Kokoro has *no* emotion control at all (not "less" — none), and
Orpheus's "emotion" is eight sound-effect tags like `<laugh>`. What
survived was Qwen3-TTS, whose `instruct` field takes free-form English
describing how to say the line. So the mapping from state to voice is
prose, not a parameter vector, and that means it must be written
carefully and pinned by tests, because a prompt is code here.

**The instruction never reaches the speaker.** It is passed
out-of-band, in a separate field from the text. A TTS that reads its own
stage direction aloud is a specific, embarrassing failure, and it is why
the emotion tag is stripped before this point rather than after.

Chatterbox (the G5 fallback) has no instruct field — it has one
`exaggeration` dial — so `exaggeration_for()` exists alongside, and both
derive from the same state rather than from each other.
"""

from __future__ import annotations

from neiro.state import EmotionLabel, NeiroState

# How each emotion sounds, at moderate strength. Deliberately describes
# DELIVERY (pace, pitch, energy, warmth) rather than naming the emotion:
# "say this angrily" tends to produce a caricature, while "clipped,
# lower, with an edge" produces something a person might actually say.
_DELIVERY: dict[EmotionLabel, str] = {
    EmotionLabel.HAPPY: "warm and bright, a little quicker than usual, smiling",
    EmotionLabel.ANGRY: "clipped and lower, tighter, with an edge — not shouting",
    EmotionLabel.SAD: "quieter and slower, falling at the end of phrases",
    EmotionLabel.RELAXED: "unhurried and soft, easy, slightly lower",
    EmotionLabel.SURPRISED: "brighter and higher, a small catch at the start",
    EmotionLabel.NEUTRAL: "even and conversational",
}

# Intensity words. The tag's digit is 0-9; three bands is the resolution
# the delivery instruction can actually carry — finer gradations produce
# no audible difference and just make the string longer.
_STRENGTH: tuple[tuple[float, str], ...] = (
    (0.35, "slightly"),
    (0.70, ""),
    (1.01, "clearly"),
)

# Never varies, so the voice stays recognisably one person across states.
# Front-loaded because an instruct string is a prompt, and what comes
# first carries most.
_IDENTITY = "A young woman talking to a friend"


def _strength_word(intensity: float) -> str:
    for ceiling, word in _STRENGTH:
        if intensity < ceiling:
            return word
    return "clearly"


def instruct_for(state: NeiroState) -> str:
    """The `instruct` string for Qwen3-TTS. Never spoken aloud.

    Kept short on purpose: long instructions dilute, and TTFA is part of
    the latency budget this project measures.
    """
    delivery = _DELIVERY[state.label]
    if state.label is EmotionLabel.NEUTRAL:
        return f"{_IDENTITY}. {delivery}."
    strength = _strength_word(state.intensity)
    qualified = f"{strength} {delivery}" if strength else delivery
    return f"{_IDENTITY}. Sounding {qualified}."


def exaggeration_for(state: NeiroState, floor: float = 0.25, ceiling: float = 0.8) -> float:
    """Chatterbox's single expressiveness dial, from the same state.

    Bounded well below 1.0 at the top: Chatterbox's own guidance is that
    high exaggeration degrades intelligibility, and a voice assistant
    that cannot be understood is worse than one that is flat. Neutral
    sits at the floor rather than at zero — a completely flat reading
    sounds synthetic even when the words are right.
    """
    if state.label is EmotionLabel.NEUTRAL:
        return floor
    return floor + (ceiling - floor) * max(0.0, min(1.0, state.intensity))
