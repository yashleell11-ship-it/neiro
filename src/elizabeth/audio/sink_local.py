"""A sink that writes to a WAV file instead of a browser.

The browser owns playback in the real system, because whoever owns the
audio clock owns the animation clock. This exists for the cases where
there is no browser: CLI testing, `elizabeth say`, and proving the loop
works before the face is built.

It keeps the same interface as the WebSocket sink, including `cancel()`,
so the orchestrator cannot tell them apart — and so barge-in's
"everything queued is dropped" behaviour is exercised even here.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLERATE = 24000


@dataclass
class LocalWavSink:
    """Accumulates chunks; writes on `close()`."""

    path: Path = Path("/tmp/elizabeth-reply.wav")
    samplerate: int = SAMPLERATE
    chunks: list[np.ndarray] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    cancelled: bool = False

    async def play(
        self,
        turn,
        pcm: np.ndarray,
        seq: int = 0,
        text: str = "",
        visemes: list[tuple[float, str, float]] | None = None,
    ) -> None:
        self.chunks.append(np.asarray(pcm, dtype=np.float32).reshape(-1))
        self.events.append(
            {
                "seq": seq,
                "text": text,
                "samples": int(np.size(pcm)),
                "visemes": len(visemes) if visemes else 0,
            }
        )
        if seq == 0:
            # The honest end of the one metric is the browser's `played`
            # callback. A file sink has no such moment, so it stamps the
            # nearest thing and says so rather than pretending.
            turn.stamp("sink_written_seq0")

    async def cancel(self, turn) -> None:
        """Barge-in: drop everything not yet written."""
        self.cancelled = True
        self.chunks.clear()

    @property
    def duration_s(self) -> float:
        return sum(len(c) for c in self.chunks) / self.samplerate

    def close(self) -> Path | None:
        if not self.chunks:
            return None
        audio = np.concatenate(self.chunks)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            # Clipping is inaudible in a level meter and very audible in
            # a speaker. Normalise rather than let it saturate.
            audio = audio / peak
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.samplerate)
            handle.writeframes((audio * 32767).astype(np.int16).tobytes())
        return self.path
