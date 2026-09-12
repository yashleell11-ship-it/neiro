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

from neiro.state import Locality, NeiroState, Turn, UserAffect


class Endpointer(Protocol):
    """Decides when the user has stopped talking. Stage 0: PTT release
    (locality is irrelevant — there is no model). Stage 2: Silero VAD +
    smart-turn v3.2.
    """

    locality: Locality

    def on_frame(self, frame: np.ndarray) -> bool:
        """Return True the instant end-of-speech is decided."""
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
        self, text: str, state: NeiroState
    ) -> AsyncIterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        """Yield ``(pcm_24k_chunk, viseme_timeline_or_None)`` per chunk."""
        ...


class Sink(Protocol):
    """Where audio is actually played. Stage 0: a local sounddevice
    stream, for CLI testing. Stage 0 (browser): the WebSocket sink —
    the browser's AudioContext owns playback from day one so barge-in
    is written once, not twice.
    """

    async def play(self, turn: Turn, pcm_chunk: np.ndarray) -> None: ...

    async def cancel(self, turn: Turn) -> None: ...
