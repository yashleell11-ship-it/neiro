"""Chatterbox — gate G5's fallback voice.

Kokoro has no emotion control at all, so the expressive voice is either
Qwen3-TTS (free-form `instruct` string) or this. Chatterbox is the
smaller, safer bet: MIT, ~0.7 GB, and one `exaggeration` dial instead of
a prompt.

**One dial, not a prompt.** `emotion/voice.py` derives it from the same
`NeiroState` that drives the face, floored at 0.25 because a completely
flat reading sounds synthetic, and ceilinged at 0.8 because Chatterbox's
own guidance is that high exaggeration costs intelligibility — and a
voice assistant that cannot be understood is worse than one that is
flat.

**No visemes.** Like every codec-style TTS, it returns audio and nothing
else, so the mouth falls back to amplitude-driven motion. That is the
whole reason Kokoro's phoneme durations were collected while they were
available.

Loaded lazily and only when selected: the import pulls a CUDA-capable
torch path, and the runtime venv deliberately has the CPU build.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neiro.config import Neiro
from neiro.emotion.voice import exaggeration_for
from neiro.state import Locality, NeiroState

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "models" / "chatterbox"
SAMPLERATE = 24000


class ChatterboxUnavailable(RuntimeError):
    """Weights or the package are missing. Raised at load, not at the
    first sentence — a voice that cannot start should stop start-up.
    """


@dataclass
class ChatterboxTts:
    """protocols.TTS. Gate G5's fallback if Qwen3-TTS does not fit."""

    locality = Locality.LOCAL_PINNED

    cfg: Neiro = field(default_factory=Neiro)
    model_dir: Path = DEFAULT_DIR
    device: str = "cpu"
    reference_wav: Path | None = None  # the frozen voice identity, once G5 picks one
    _model: object = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise ChatterboxUnavailable(
                f"No Chatterbox weights at {self.model_dir} — run "
                "`neiro fetch-models --only chatterbox`."
            )
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise ChatterboxUnavailable(
                "chatterbox-tts is not installed. It is deliberately not a runtime "
                "dependency until gate G5 picks it — installing it pulls a CUDA torch "
                "path, and this venv keeps the CPU build so ctranslate2's cuBLAS 12 "
                "stays the only CUDA runtime in the process."
            ) from exc
        self._model = ChatterboxTTS.from_local(str(self.model_dir), device=self.device)

    def warm(self) -> float:
        """Pay the first-call cost at start-up, like every other runtime."""
        import time

        started = time.perf_counter()
        self.load()
        return time.perf_counter() - started

    async def synth(
        self, text: str, state: NeiroState | None = None
    ) -> AsyncIterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        """Yield `(pcm_24k, None)`. The `None` is the absence of visemes,
        stated rather than implied — the caller must fall back to
        amplitude-driven mouth motion.
        """
        import asyncio

        self.load()
        exaggeration = exaggeration_for(state) if state else 0.25

        def run() -> np.ndarray:
            kwargs: dict = {"exaggeration": exaggeration}
            if self.reference_wav is not None:
                kwargs["audio_prompt_path"] = str(self.reference_wav)
            wav = self._model.generate(text, **kwargs)
            audio = np.asarray(getattr(wav, "cpu", lambda: wav)(), dtype=np.float32).reshape(-1)
            return audio

        loop = asyncio.get_running_loop()
        # Synchronous and CPU-bound; running it inline would block the
        # event loop that is also feeding audio to the browser.
        yield await loop.run_in_executor(None, run), None
