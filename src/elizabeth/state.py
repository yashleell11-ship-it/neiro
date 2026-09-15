"""The Turn is the spine of the whole pipeline.

Every stage (capture, VAD/endpoint, STT, affect, LLM, TTS, tools) is
written against these types from Stage 0 onward, even though most fields
are inert until later stages fill them in. That is deliberate: Stage 2's
barge-in and Stage 3's tools become additions to this object, not a
rewrite of it.

Two rules that must never be broken (see CLAUDE.md):
  - `user_affect` (what she heard) and `elizabeth_state` (what she feels)
    are never assigned to each other.
  - The internal emotion state is continuous (floats); a discrete label
    is only ever a *view* onto it, for the face.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class EmotionLabel(StrEnum):
    """Exactly the VRM 1.0 normative expression preset names.

    VRM 1.0 spells these lowercase — 'relaxed', not 'calm' — and the
    mouth-shape presets (not modeled here) are aa/ih/ou/ee/oh, NOT the
    VRM 0.x A/I/U/E/O. Getting this spelling wrong makes
    expressionManager.setValue() silently do nothing.
    """

    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    RELAXED = "relaxed"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"


# Fixed valence/arousal lookup for each label. Used because TTS prosody
# control and VRM blendshape weights both want continuous floats, not a
# bare categorical — see docs/CORRECTIONS.md.
_VALENCE_AROUSAL: dict[EmotionLabel, tuple[float, float]] = {
    EmotionLabel.HAPPY: (0.8, 0.6),
    EmotionLabel.ANGRY: (-0.6, 0.8),
    EmotionLabel.SAD: (-0.7, -0.4),
    EmotionLabel.RELAXED: (0.4, -0.6),
    EmotionLabel.SURPRISED: (0.2, 0.9),
    EmotionLabel.NEUTRAL: (0.0, 0.0),
}


@dataclass(frozen=True)
class ElizabethState:
    """What SHE feels. Output side. Parsed from her own leading
    ``<e:LABEL:D>`` tag (see llm/emotion_tag.py). Fans out to three
    consumers: the VRM face, the TTS ``instruct`` string, and the next
    turn's prompt.
    """

    label: EmotionLabel = EmotionLabel.NEUTRAL
    intensity: float = 0.5  # 0..1, the tag's digit / 9
    valence: float = 0.0
    arousal: float = 0.0

    @classmethod
    def from_label(cls, label: EmotionLabel, intensity: float) -> ElizabethState:
        v, a = _VALENCE_AROUSAL[label]
        return cls(label=label, intensity=intensity, valence=v, arousal=a)


NEUTRAL_STATE = ElizabethState()


@dataclass(frozen=True)
class UserAffect:
    """What we HEARD. Input side. Inert (``NONE``) until gate G3b (Stage 0
    Task 13) passes and Lane A ships in Stage 1. Never merge this with
    ``ElizabethState`` — sharing one variable is how an assistant ends up
    reading its own TTS back as the user's mood.
    """

    arousal_z: float = 0.0
    valence_z: float = 0.0
    confidence: float = 0.0  # 0 = "say nothing about this"
    events: tuple[str, ...] = ()  # e.g. ("laughter",)
    baseline_n: int = 0  # utterances the rolling baseline has seen


UserAffect.NONE = UserAffect()  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ToolCall:
    """Stage 3. The accumulator exists from Stage 0 and never fires
    until the tool registry lands — see llm/openai_compat.py.
    """

    name: str
    args: dict
    index: int


class Tier(StrEnum):
    """Where this turn's TIERABLE providers ran. Snapshotted at turn
    start and never changed mid-turn, so a promotion or demotion can
    only ever happen between turns.

    Corrected 2026-09-13 (docs/DECISIONS.md): the 3090 Ti box is not one
    remote place, it is two very different links to the same machine.
    """

    LOCAL = "local"  # the machine Yash is sitting at (the laptop, for now)
    LAN = "lan"  # the 3090 Ti over ethernet on the same router — home
    TUNNEL = "tunnel"  # the 3090 Ti via Cloudflare Access — hostel→home, gated on T17b


class Locality(StrEnum):
    """Declared by every provider Protocol (see protocols.py).

    LOCAL_PINNED providers never leave the machine Yash is sitting at,
    as an architectural rule rather than a default: capture, VAD,
    endpointing, affect, the audio sink, and every tool. The mic and the
    desktop are wherever he is, by definition.

    LAN_TIERABLE is STT alone: shipping 16 kHz audio for every streaming
    partial is fine over ethernet and breaks the sub-second design over
    a tunnel. TIERABLE (LLM, TTS) goes anywhere the resolver says.

    The rule lives in `allows()` so the resolver enforces it in code —
    not in a comment someone reads once.
    """

    LOCAL_PINNED = "local_pinned"
    LAN_TIERABLE = "lan_tierable"
    TIERABLE = "tierable"

    def allows(self, tier: Tier) -> bool:
        """May a provider with this locality run on `tier`?"""
        if self is Locality.LOCAL_PINNED:
            return tier is Tier.LOCAL
        if self is Locality.LAN_TIERABLE:
            return tier in (Tier.LOCAL, Tier.LAN)
        return True


@dataclass
class Turn:
    """One utterance, start to end. Created at speech-start, mutated
    exactly once per pipeline stage, carried by reference through every
    queue.
    """

    id: int
    t0_speech_start: float = field(default_factory=time.monotonic)
    t_endpoint: float | None = None

    audio: np.ndarray | None = None  # float32 16 kHz mono, the whole utterance
    transcript: str | None = None
    partial: str | None = None  # Stage 2: streaming partial transcript

    user_affect: UserAffect = field(default_factory=lambda: UserAffect.NONE)
    elizabeth_state: ElizabethState = field(default_factory=lambda: NEUTRAL_STATE)
    tool_calls: list[ToolCall] = field(default_factory=list)

    tier: Tier = Tier.LOCAL  # snapshotted at turn start; never changes mid-turn
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    timeline: dict[str, float] = field(default_factory=dict)

    def stamp(self, name: str) -> None:
        """Record a perf_counter timestamp for this stage boundary.
        metrics.py reads this dict to build the one metric and the
        per-stage waterfall.
        """
        self.timeline[name] = time.perf_counter()

    @classmethod
    def new(cls, turn_id: int = 0) -> Turn:
        return cls(id=turn_id)
