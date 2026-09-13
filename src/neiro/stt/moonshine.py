"""Moonshine — the CPU speech recogniser, for the Stage 2 A/B.

distil-whisper is the Stage 0 choice and it is good: 141 ms on the GPU,
measured. Moonshine's argument is different — it runs on the CPU at
**zero VRAM**, and gate G2's no-go ladder starts with "move STT off the
GPU" precisely because the LLM, the TTS and a WebGL browser tab all want
the same 7730 MiB.

**Its cost is not free, and the plan says so.** The research put
Moonshine Medium Streaming at ~269 ms *after* endpointing — additive,
not hidden. So this is not a drop-in improvement; it is a trade of
latency for VRAM, and Stage 2 decides it on Yash's own WER set with a
gate of "no worse than 1.5x the distil baseline".

Same `protocols.STT` surface as `faster_whisper.py`, so the A/B is a
config change and the orchestrator never learns which one it has.

**English-only.** The checkpoints this module names (`moonshine/medium`
and its siblings) were trained on English speech alone; there is no
Hindi one to load. So `cfg.stt.language = "hi"` is refused at
construction — loudly, before the first utterance, with
`MoonshineEnglishOnly` — rather than letting Hindi audio come back as
English-shaped nonsense that the WER set would then dutifully score.
"auto" means English here: with one language there is nothing to
detect, and that is the half of the brief Moonshine can be measured on
at all. The Hindi half of the A/B is faster-whisper's alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neiro.config import Neiro
from neiro.state import Locality

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLERATE = 16000

# The same silence hallucinations faster_whisper.py blocks. Whisper-family
# models confidently transcribe an empty room as "Thank you"; Moonshine is
# a different architecture but trained on similar data, so the floor is
# applied to both rather than assumed to be unnecessary here.
_HALLUCINATIONS = frozenset(
    {"thank you", "thanks for watching", "you", ".", "bye", "okay", "um", "uh"}
)


class MoonshineUnavailable(RuntimeError):
    """Weights or the package are missing."""


class MoonshineEnglishOnly(MoonshineUnavailable):
    """Asked for a language these checkpoints cannot transcribe.

    A subclass of `MoonshineUnavailable` on purpose: to the caller this
    is the same situation — this recogniser cannot serve this config —
    and `transcribe()` already lets that class through untouched instead
    of swallowing it into a silent "".
    """


@dataclass
class MoonshineStt:
    """protocols.STT on CPU. LAN_TIERABLE like the other recogniser —
    audio may cross ethernet, never a tunnel.

    Refuses `cfg.stt.language = "hi"` in `__post_init__`, so a daemon
    configured for Hindi with Moonshine selected fails at startup and not
    on his first sentence — see the module docstring.
    """

    locality = Locality.LAN_TIERABLE

    cfg: Neiro = field(default_factory=Neiro)
    model_name: str = "moonshine/medium"
    _model: object = None

    def __post_init__(self) -> None:
        if self.cfg.stt.language == "hi":
            raise MoonshineEnglishOnly(
                f"stt.language = {self.cfg.stt.language!r}, but {self.model_name} is an "
                "English-only checkpoint: Moonshine has no Hindi model. Use "
                'faster-whisper for Hindi, or set stt.language to "en" / "auto" '
                "(both mean English here)."
            )

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import moonshine_onnx  # noqa: F401
        except ImportError as exc:
            raise MoonshineUnavailable(
                "useful-moonshine-onnx is not installed. It is not a runtime "
                "dependency until Stage 2's A/B picks it — distil-whisper is the "
                "Stage 0 recogniser and already measured at 141 ms."
            ) from exc
        self._model = self.model_name

    def warm(self) -> float:
        """One throwaway transcription. Gate G1's lesson, applied to
        every runtime since: the first call costs far more than the rest.
        """
        import time

        started = time.perf_counter()
        self.load()
        self._run(np.zeros(SAMPLERATE, dtype=np.float32))
        return time.perf_counter() - started

    def _run(self, pcm: np.ndarray) -> str:
        import moonshine_onnx

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        text = moonshine_onnx.transcribe(audio, self._model)
        return (text[0] if isinstance(text, list) else str(text)).strip()

    async def transcribe(self, pcm_16k: np.ndarray) -> str:
        """Transcribe, or return `""` for "nothing usable was said".

        Empty string rather than a guess, for the same reason as
        faster_whisper: answering silence is how an assistant replies to
        an empty room, and the orchestrator already treats "" as a
        rejected turn.
        """
        import asyncio

        self.load()
        loop = asyncio.get_running_loop()
        try:
            # CPU-bound and synchronous; inline would block the loop that
            # is also feeding audio to the browser.
            text = await loop.run_in_executor(None, self._run, pcm_16k)
        except MoonshineUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed turn is not a dead daemon
            log.warning("Moonshine failed on one utterance: %s", exc)
            return ""
        return "" if text.lower().strip(" .,!?") in _HALLUCINATIONS else text
