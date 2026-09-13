"""Kokoro — the Stage 0 voice, and the source of free lip-sync.

Deliberately CPU: torch here is the `+cpu` build (see pyproject), because
ctranslate2 already loads cuBLAS 12 through `LD_LIBRARY_PATH` and a
second CUDA runtime in the same process is how the Gate G1 problem comes
back. Kokoro is 82M parameters; the GPU is fully committed to the LLM and
STT, and this costs it nothing.

**What it is for.** Not quality, and certainly not emotion — Kokoro has
*no* emotion control at all, which is the whole reason gate G5 exists.
It is here because it returns per-phoneme durations, and those are a
viseme timeline for free. Every codec-style TTS that might replace it
(Qwen3-TTS, Chatterbox) returns audio and nothing else, leaving the mouth
to be driven from amplitude. Collecting real visemes now is what makes
Stage 1's face cheap.

**Time to first audio scales with the first sentence**, which is why the
character prompt asks her to open with a short clause and why the
chunker exists. `synth()` yields per sentence rather than per reply so
the first words reach the browser while the rest is still being made.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np

from neiro.config import Neiro
from neiro.state import Locality, NeiroState
from neiro.tts.visemes import timeline

SAMPLERATE = 24000  # Kokoro's native rate, and the rate the browser expects
# Kokoro's duration head emits frames at 80 per second.
FRAMES_PER_SECOND = 80.0


class KokoroTts:
    """protocols.TTS. LOCAL_PINNED in practice — it is CPU and tiny, so
    there is nothing to gain by moving it, and audio bytes are the one
    thing not worth sending over a link.
    """

    locality = Locality.LOCAL_PINNED

    def __init__(self, cfg: Neiro | None = None, voice: str = "af_heart") -> None:
        self._cfg = cfg or Neiro()
        self._voice = voice
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            # lang_code 'a' is American English. Kokoro downloads its own
            # weights on first use; `neiro fetch-models` pre-fetches them
            # so a cold start is not a surprise.
            self._pipeline = KPipeline(lang_code="a")
        return self._pipeline

    def warm(self) -> float:
        """One throwaway synthesis. Same lesson as ctranslate2, librosa
        and onnxruntime: the first call into a runtime costs far more
        than the rest, and it must not be paid on his first sentence.
        """
        started = time.perf_counter()
        for _ in self._synth_sync("Hi."):
            break
        return time.perf_counter() - started

    def _synth_sync(
        self, text: str
    ) -> Iterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        pipeline = self._load()
        for result in pipeline(text, voice=self._voice):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)

            visemes = None
            phonemes = getattr(result, "phonemes", None)
            durations = getattr(getattr(result, "output", None), "pred_dur", None)
            if phonemes is not None and durations is not None:
                # pred_dur is in 80-per-second frames, not seconds.
                seconds = [float(d) / FRAMES_PER_SECOND for d in np.asarray(durations).reshape(-1)]
                visemes = timeline(phonemes, seconds)
            yield pcm, visemes

    async def synth(
        self, text: str, state: NeiroState | None = None
    ) -> AsyncIterator[tuple[np.ndarray, list[tuple[float, str, float]] | None]]:
        """Yield `(pcm_24k, visemes_or_None)` per sentence.

        `state` is accepted and ignored, on purpose. Kokoro has no
        emotion control, and silently accepting the argument keeps the
        Protocol honest: swapping in an expressive engine at gate G5
        changes this file and nothing else.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        # Kokoro is synchronous and CPU-bound. Running it inline would
        # block the event loop that is also feeding audio to the browser.
        for chunk in await loop.run_in_executor(None, lambda: list(self._synth_sync(text))):
            yield chunk


def preflight(models_dir: Path | None = None) -> str:
    """Say whether Kokoro can run at all, without synthesising."""
    try:
        import kokoro  # noqa: F401
        import torch
    except ImportError as exc:
        return f"missing dependency: {exc.name}"
    if "+cpu" not in torch.__version__ and torch.cuda.is_available():
        return (
            f"torch {torch.__version__} is a CUDA build. Kokoro must use the CPU "
            "build here — a second CUDA runtime alongside ctranslate2's cuBLAS 12 "
            "is exactly the Gate G1 failure."
        )
    return "ok"
