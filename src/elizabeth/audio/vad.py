"""Silero VAD — is someone speaking right now?

Two jobs, and they are not the same job.

**Endpointing (Stage 2).** VAD is the cheap *trigger*, not the decision.
It says "he stopped making noise"; smart-turn then decides whether he
finished a *turn*. Conflating the two is what cuts people off at "I want
to go to... uh...". So `SpeechGate` deliberately exposes the raw
transition and lets the caller decide what it means.

**Barge-in (Stage 2).** While Elizabeth is speaking, a much stricter gate:
higher probability, sustained longer, and armed only after a dead zone,
so the first syllable of her own reply cannot interrupt her.

Three things Silero is strict about, all of which fail quietly:

  - **512 samples at 16 kHz. Exactly.** v5 dropped support for other
    frame sizes; feeding 1024 gives wrong numbers rather than an error.
  - **It is stateful.** A 2x1x128 RNN state must be carried between
    frames in order. Dropping it makes every frame look like the start
    of speech, which mostly reads as "works, but jumpy".
  - **`reset()` between utterances**, or the tail of the last one biases
    the start of the next.

CPU ONNX runtime on purpose: ~1 ms per frame, no VRAM, and the GPU is
fully committed (see the VRAM ledger).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from elizabeth.config import Elizabeth

REPO_ROOT = Path(__file__).resolve().parents[3]


class VadFrameSizeError(ValueError):
    """Raised when a frame is not exactly the size Silero requires."""


class SileroVad:
    """Speech probability per 32 ms frame."""

    def __init__(self, cfg: Elizabeth | None = None, model_path: Path | None = None) -> None:
        self._cfg = cfg or Elizabeth()
        path = model_path or (REPO_ROOT / self._cfg.vad.model_path)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {path} — run "
                "`elizabeth fetch-models --only silero-vad`."
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread each: this runs per 32 ms frame alongside capture and
        # (later) STT. Letting ORT spawn a thread pool for a 1 ms model
        # costs more in scheduling than it saves.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    @property
    def frame_samples(self) -> int:
        return self._cfg.vad.frame_samples

    def reset(self) -> None:
        """Clear the RNN state. Call between utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for one frame.

        Rejects a wrong-sized frame loudly. Silero v5 accepts only 512
        samples at 16 kHz and returns plausible-looking nonsense for
        anything else — a silent failure this project cannot afford in
        the component that decides when Yash has finished talking.
        """
        audio = np.asarray(frame, dtype=np.float32).reshape(-1)
        if audio.size != self.frame_samples:
            raise VadFrameSizeError(
                f"Silero VAD needs exactly {self.frame_samples} samples at 16 kHz "
                f"(32 ms), got {audio.size}. Other sizes return plausible nonsense "
                "rather than an error."
            )
        output, self._state = self._session.run(
            None, {"input": audio[np.newaxis, :], "state": self._state, "sr": self._sr}
        )
        return float(output[0][0])

    def warm(self) -> float:
        """One throwaway inference, so the first real frame is not slow.

        Same lesson as ctranslate2 (Gate G1) and librosa: the first call
        into a runtime costs far more than the rest.
        """
        import time

        started = time.perf_counter()
        self.probability(np.zeros(self.frame_samples, dtype=np.float32))
        self.reset()
        return time.perf_counter() - started


@dataclass
class SpeechGate:
    """Hysteresis over raw VAD probabilities.

    Entering speech is fast; leaving it is slow. That asymmetry is
    deliberate and it is the same principle as the expression blender:
    cutting someone off mid-sentence is much worse than a little trailing
    silence.

    Reports *transitions*, so the caller decides what they mean. In
    Stage 2 a `stopped` transition is the trigger that wakes smart-turn,
    not the decision that the turn is over.
    """

    cfg: Elizabeth = field(default_factory=Elizabeth)
    frame_ms: float = 32.0
    speaking: bool = False
    _above: float = 0.0  # ms of consecutive above-threshold audio
    _below: float = 0.0

    def reset(self) -> None:
        self.speaking = False
        self._above = self._below = 0.0

    def update(self, probability: float) -> str | None:
        """Feed one frame's probability. Returns "started", "stopped", or
        None when nothing changed.
        """
        if probability >= self.cfg.vad.threshold:
            self._above += self.frame_ms
            self._below = 0.0
        else:
            self._below += self.frame_ms
            self._above = 0.0

        if not self.speaking and self._above >= self.cfg.vad.min_speech_ms:
            self.speaking = True
            return "started"
        if self.speaking and self._below >= self.cfg.vad.min_silence_ms:
            self.speaking = False
            return "stopped"
        return None


@dataclass
class BargeInDetector:
    """Did Yash start talking over her?

    Stricter than `SpeechGate` on every axis, because a false positive
    cuts Elizabeth off mid-sentence and a false negative just means he has to
    repeat himself. The dead zone exists because her own first syllable
    reaches the mic before any echo canceller has adapted.
    """

    cfg: Elizabeth = field(default_factory=Elizabeth)
    frame_ms: float = 32.0
    _playing_ms: float = 0.0
    _speech_ms: float = 0.0

    def start_speaking(self) -> None:
        """Elizabeth has begun to talk. Re-arms the dead zone."""
        self._playing_ms = 0.0
        self._speech_ms = 0.0

    def stop_speaking(self) -> None:
        self._playing_ms = 0.0
        self._speech_ms = 0.0

    @property
    def armed(self) -> bool:
        return self._playing_ms >= self.cfg.vad.bargein_dead_zone_ms

    def update(self, probability: float) -> bool:
        """One frame while she is speaking. True means: stop her now."""
        self._playing_ms += self.frame_ms
        if not self.armed:
            # Her own first syllable must never count as an interruption.
            self._speech_ms = 0.0
            return False
        if probability >= self.cfg.vad.bargein_probability:
            self._speech_ms += self.frame_ms
        else:
            self._speech_ms = 0.0
        return self._speech_ms >= self.cfg.vad.bargein_speech_ms
