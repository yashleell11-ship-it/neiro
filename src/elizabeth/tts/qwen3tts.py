"""Qwen3-TTS — gate G5's primary candidate, and the reason G5 exists.

Kokoro has **no** emotion control. Not less of it — none. Orpheus's is
eight sound-effect tags and it is HF-gated, which breaks "no API keys".
What survived the research pass was Qwen3-TTS, whose `instruct` field
takes free-form English describing *how* to say the line — which is why
`emotion/voice.py` writes prose rather than setting a parameter.

**The instruction is out-of-band and must never be spoken.** It travels
in a separate field from the text. A TTS reading its own stage direction
aloud is a specific, embarrassing failure, and it is why the `<e:>` tag
is stripped upstream rather than here.

**G5 is a real gate with a number**: TTFA under 500 ms with the LLM
resident, and it must fit in what the VRAM ledger leaves — likely only
with STT moved to CPU, which is ladder step 1. Until it is measured,
Kokoro is the voice and this is a candidate.

**No visemes**, like every codec-style model. Kokoro's free phoneme
durations are the thing being given up, and the mouth falls back to
amplitude-driven motion.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from elizabeth.config import Elizabeth
from elizabeth.emotion.voice import instruct_for
from elizabeth.state import ElizabethState, Locality

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "models" / "qwen3-tts-1.7b"
SAMPLERATE = 24000

# The voice identity is a ONE-WAY DOOR: once she has a voice, changing it
# makes her a different character. G5 freezes a speaker and a rendered
# sample as a versioned asset, and this is where that choice lands.
DEFAULT_SPEAKER = "Ono_Anna"


class Qwen3TtsUnavailable(RuntimeError):
    """Weights or the package are missing."""


@dataclass
class Qwen3Tts:
    """protocols.TTS. The expressive candidate."""

    locality = Locality.LOCAL_PINNED

    cfg: Elizabeth = field(default_factory=Elizabeth)
    model_dir: Path = DEFAULT_DIR
    speaker: str = DEFAULT_SPEAKER
    device: str = "cuda"
    chunk_size: int = 8  # streaming granularity; G5 benchmarks 4 vs 8
    _model: object = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise Qwen3TtsUnavailable(
                f"No Qwen3-TTS weights at {self.model_dir} — run "
                "`elizabeth fetch-models --only qwen3-tts-1.7b`."
            )
        try:
            from faster_qwen3_tts import Qwen3TTS
        except ImportError as exc:
            raise Qwen3TtsUnavailable(
                "faster-qwen3-tts is not installed. Deliberately not a runtime "
                "dependency until gate G5 picks it: it pulls a CUDA torch, and this "
                "venv keeps the CPU build so ctranslate2's cuBLAS 12 stays the only "
                "CUDA runtime in the process. G5 benchmarks it in its own venv."
            ) from exc
        self._model = Qwen3TTS(str(self.model_dir), device=self.device)

    def warm(self) -> float:
        import time

        started = time.perf_counter()
        self.load()
        return time.perf_counter() - started

    async def synth(
        self, text: str, state: ElizabethState | None = None
    ) -> AsyncIterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        """Yield `(pcm_24k, None)` per streamed chunk.

        `instruct` carries the emotion and is never part of `text` — the
        separation is the whole safety property, so the two are passed as
        distinct arguments rather than concatenated anywhere.
        """
        import asyncio

        self.load()
        instruct = instruct_for(state) if state else None

        def run() -> list[np.ndarray]:
            chunks = self._model.generate_stream(
                text=text,
                speaker=self.speaker,
                instruct=instruct,
                chunk_size=self.chunk_size,
            )
            return [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]

        loop = asyncio.get_running_loop()
        for chunk in await loop.run_in_executor(None, run):
            # None: no visemes. Stated, not implied — the caller must
            # fall back to amplitude-driven mouth motion.
            yield chunk, None
