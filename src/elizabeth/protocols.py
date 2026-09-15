"""Interfaces for every swappable provider.

Writing these before any implementation is what makes Stage 2's SER
model, Stage 2's expressive TTS, and Stage 4's remote LLM tier additions
rather than rewrites: something already implements the Protocol, and the
orchestrator never changes when a provider is swapped behind it.

Every Protocol carries a ``locality`` class attribute (see state.Locality).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from elizabeth.state import ElizabethState, Locality, Turn, UserAffect


class Endpointer(Protocol):
    """Decides whether the user has stopped talking. Stage 0: PTT release
    (no model, nothing to ask). Stage 2: Silero VAD triggers, smart-turn
    v3.2 decides — this is the decider's shape. LOCAL_PINNED: it runs
    where the microphone is.
    """

    locality: Locality

    def is_complete(self, pcm: np.ndarray, waited_s: float = 0.0) -> tuple[bool, float]:
        """`(decided, probability)` for the speech captured so far.
        `waited_s` is how long since the VAD trigger, so the caller's
        max-delay rule can be applied inside one place.
        """
        ...


class STT(Protocol):
    """Speech to text. LAN_TIERABLE: may run on the 3090 Ti over
    ethernet, never over the tunnel — even 60 ms of RTT on every
    streaming partial breaks the sub-second design. Whether LAN STT
    actually beats local STT is a measurement (T17a), not a default.
    """

    locality: Locality

    async def transcribe(self, pcm_16k: np.ndarray) -> str: ...


class AffectProvider(Protocol):
    """What we hear in the user's voice. LOCAL_PINNED. ``null.py``
    ships in Stage 0 and always returns ``UserAffect.NONE`` — the
    orchestrator is unaware whether it's talking to the null provider or
    a real one.
    """

    locality: Locality

    async def observe(self, pcm_window_16k: np.ndarray) -> UserAffect: ...

    def commit_utterance(self) -> bool:
        """End of a turn he really spoke: fold what was heard into his
        baseline. Once per utterance, never per window. Returns True if
        drift detection reset the baseline.
        """
        ...

    def discard_utterance(self) -> None:
        """End of a turn that is not a sample of how he normally sounds —
        a rejected transcript (the STT floor said nothing usable was
        said) or an interrupted one. Drops what observe() staged so it
        can neither enter the baseline nor decide the next turn's band.
        """
        ...


class LLM(Protocol):
    """One OpenAI-compatible client shape for every tier — promoting to
    the 3090 Ti (over LAN or the tunnel) is a ``base_url`` change, not a
    new code path.
    """

    locality: Locality

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator[dict]:
        """Yield OpenAI-style ``delta`` fragments."""
        ...


class TTS(Protocol):
    """Text to speech. Returns audio plus a viseme timeline — ``None``
    when the engine gives no alignment (every codec model except
    Kokoro), so the caller falls back to amplitude-driven mouth motion.
    """

    locality: Locality

    async def synth(
        self, text: str, state: ElizabethState
    ) -> AsyncIterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        """Yield ``(pcm_24k_chunk, viseme_timeline_or_None)`` per chunk."""
        ...


class Sink(Protocol):
    """Where audio is actually played. Stage 0: a local sounddevice
    stream, for CLI testing. Stage 0 (browser): the WebSocket sink —
    the browser's AudioContext owns playback from day one so barge-in
    is written once, not twice.
    """

    async def play(
        self,
        turn: Turn,
        pcm: np.ndarray,
        seq: int = 0,
        text: str = "",
        visemes: list[tuple[float, str, float]] | None = None,
    ) -> None:
        """One 24 kHz chunk. `seq` 0 is the chunk whose playback ends the
        one metric; `text` and `visemes` are what the face needs to move
        its mouth in time with it.
        """
        ...

    async def cancel(self, turn: Turn) -> None: ...
