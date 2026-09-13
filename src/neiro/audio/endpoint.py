"""smart-turn v3.2 — has he finished a turn, or just paused?

**VAD answers a different question.** It says "he stopped making noise",
which is true a dozen times inside one sentence. Acting on it is what
cuts people off at *"I want to go to… uh… the library"*. So `vad.py`
is the cheap trigger and this is the decision: when the speech gate
reports a stop, this looks at the last 8 seconds and says whether the
turn is actually over.

**Asymmetric on purpose, again.** The completion threshold sits above
0.5 — letting someone finish costs a moment, cutting them off costs the
whole turn and the goodwill with it. The same reasoning as the speech
gate's hysteresis and the expression blender's fall time.

**And a hard ceiling.** `max_wait_s` ends the turn regardless. A model
that never fires would otherwise hang the conversation, which is worse
than one early cut.

BSD-2 (pipecat-ai), standalone ONNX, ~8.7 MB, CPU — no VRAM, and the
GPU is fully committed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neiro.config import Neiro

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLERATE = 16000


class EndpointerUnavailable(RuntimeError):
    """The ONNX model is missing."""


def log_mel(pcm: np.ndarray, cfg: Neiro, samplerate: int = SAMPLERATE) -> np.ndarray:
    """80-bin log-mel over exactly `n_frames`, shaped (1, 80, 800).

    The window is the most RECENT `context_seconds`, not the first: the
    end of an utterance is what decides whether it ended. Shorter audio
    is left-padded so the real speech still sits at the right-hand edge,
    where the model expects it.
    """
    import librosa

    want = int(cfg.endpoint.context_seconds * samplerate)
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    audio = audio[-want:] if audio.size >= want else np.pad(audio, (want - audio.size, 0))

    # WHISPER's feature pipeline exactly, because smart-turn is built on
    # Whisper's encoder and was trained on its inputs. A generic
    # `power_to_db(ref=np.max)` produced a CONSTANT 0.729 for every clip
    # AND for silence — the model saw features it had never been trained
    # on and fell back to its prior, which looks like a working model
    # returning a plausible number.
    hop = max(1, want // cfg.endpoint.n_frames)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=samplerate, n_mels=cfg.endpoint.n_mels, n_fft=400, hop_length=hop, center=True
    )
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # Trim or pad to exactly n_frames — librosa's centring can differ by
    # one, and the ONNX shape is fixed.
    frames = cfg.endpoint.n_frames
    mel = (
        mel[:, :frames]
        if mel.shape[1] >= frames
        else np.pad(mel, ((0, 0), (0, frames - mel.shape[1])))
    )
    return mel[np.newaxis, :, :].astype(np.float32)


@dataclass
class SmartTurnEndpointer:
    """Probability that the utterance is complete."""

    cfg: Neiro = field(default_factory=Neiro)
    model_path: Path | None = None
    _session: object = None

    def load(self) -> None:
        if self._session is not None:
            return
        path = self.model_path or (REPO_ROOT / self.cfg.endpoint.model_path)
        if not Path(path).exists():
            raise EndpointerUnavailable(
                f"smart-turn model not found at {path} — run "
                "`neiro fetch-models --only smart-turn-v3`."
            )
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread: this runs once per candidate endpoint, alongside
        # capture and VAD. A thread pool for a ~13 ms model costs more in
        # scheduling than it saves.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def warm(self) -> float:
        """One throwaway inference. Same trap as every other runtime here."""
        import time

        started = time.perf_counter()
        self.load()
        self.probability(np.zeros(SAMPLERATE, dtype=np.float32))
        return time.perf_counter() - started

    def probability(self, pcm: np.ndarray) -> float:
        """Probability in [0, 1] that the turn is complete."""
        self.load()
        features = log_mel(pcm, self.cfg)
        logits = self._session.run(None, {"input_features": features})[0]
        return float(1.0 / (1.0 + np.exp(-float(np.asarray(logits).reshape(-1)[0]))))

    def is_complete(self, pcm: np.ndarray, waited_s: float = 0.0) -> tuple[bool, float]:
        """`(complete, probability)` for this candidate endpoint.

        `waited_s` is how long the caller has already been waiting since
        speech began. Past `max_wait_s` the answer is yes regardless — a
        model that never fires must not be able to hang the conversation.
        """
        if waited_s >= self.cfg.endpoint.max_wait_s:
            return True, 1.0
        try:
            p = self.probability(pcm)
        except EndpointerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — fall back to VAD's opinion
            log.warning("smart-turn failed (%s); treating the VAD stop as the endpoint", exc)
            return True, 0.0
        return p >= self.cfg.endpoint.complete_threshold, p
