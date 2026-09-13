"""Lane B: the trained speech-emotion model, at runtime.

`training/recipes/ser_train.py` produces a checkpoint that predicts
valence and arousal on the circumplex. This loads it and presents it as
an `AffectProvider`, interchangeable with Lane A's prosody and with the
null provider — so turning Lane B on is a config flip, not a code path.

**Its output is not comparable to Lane A's until it is baselined.** The
model predicts absolute circumplex coordinates learned from actors;
Lane A produces z-scores against Yash's own normal. `fusion.py` holds
the calibration that puts them on one scale, and this provider deliberately
returns the *raw* prediction plus the calibrated z-score rather than
pretending they are the same number.

**Why CPU is the default.** The GPU is fully committed (see the VRAM
ledger) and this runs on a rolling window *while the user is still
speaking*, so a slower-but-free inference costs the turn budget nothing.
`device="cuda"` is available for the box tier where VRAM is not scarce.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neiro.affect.fusion import LaneBCalibration
from neiro.config import Neiro
from neiro.state import Locality, UserAffect

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "ser-lane-b" / "best.pt"
DEFAULT_ENCODER = REPO_ROOT / "models" / "w2v-bert-2.0"
SAMPLERATE = 16000


class SerUnavailable(RuntimeError):
    """The checkpoint or its encoder is missing. Raised at construction,
    not at the first utterance — a model that cannot load should stop
    start-up, not fail silently mid-conversation.
    """


@dataclass
class SerAffectProvider:
    """protocols.AffectProvider backed by the trained regressor.

    LOCAL_PINNED: it reads the microphone's audio, and audio is the one
    thing not worth sending over a link.
    """

    locality = Locality.LOCAL_PINNED

    cfg: Neiro = field(default_factory=Neiro)
    checkpoint: Path = DEFAULT_CHECKPOINT
    encoder: Path = DEFAULT_ENCODER
    device: str = "cpu"
    calibration: LaneBCalibration = field(default_factory=LaneBCalibration)
    _model: object = None
    _processor: object = None
    _last: tuple[float, float] | None = None

    def load(self) -> None:
        """Bring up the model. Raises `SerUnavailable` if it cannot."""
        if self._model is not None:
            return
        if not self.checkpoint.exists():
            raise SerUnavailable(
                f"No Lane B checkpoint at {self.checkpoint}. Train one:\n"
                "  cd training && uv run python recipes/ser_train.py --corpora crema-d ravdess rasa"
            )
        if not self.encoder.exists():
            raise SerUnavailable(
                f"Encoder missing at {self.encoder} — run `neiro fetch-models --only w2v-bert-2.0`."
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise SerUnavailable("torch is not installed in the runtime venv") from exc

        import sys

        sys.path.insert(0, str(REPO_ROOT / "training" / "recipes"))
        from ser_train import SerRegressor  # the exact architecture that was trained
        from transformers import AutoFeatureExtractor

        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        unfreeze = (state.get("args") or {}).get("unfreeze", 4)
        model = SerRegressor(self.encoder, unfreeze_layers=unfreeze)
        model.load_state_dict(state["model"])
        model.eval().to(self.device)
        self._model = model
        self._processor = AutoFeatureExtractor.from_pretrained(str(self.encoder))

    def warm(self) -> float:
        """Load and run one throwaway inference. Same reason as every
        other `warm()` here: the first call into a runtime costs far more
        than the rest, and it must not land on his first sentence.
        """
        import time

        started = time.perf_counter()
        self.load()
        self._predict(np.zeros(SAMPLERATE, dtype=np.float32))
        return time.perf_counter() - started

    def _predict(self, pcm: np.ndarray) -> tuple[float, float]:
        """Raw circumplex prediction: `(valence, arousal)` in [-1, 1]."""
        import torch

        features = self._processor(
            [np.asarray(pcm, dtype=np.float32).reshape(-1)],
            sampling_rate=SAMPLERATE,
            return_tensors="pt",
        )["input_features"].to(self.device)
        with torch.no_grad():
            out = self._model(features).float().cpu().numpy()[0]
        return float(out[0]), float(out[1])

    async def observe(self, pcm_window_16k: np.ndarray) -> UserAffect:
        """One rolling window. Returns a z-scored reading, or NONE while
        the calibration is still cold.
        """
        audio = np.asarray(pcm_window_16k, dtype=np.float32).reshape(-1)
        if audio.size < int(self.cfg.affect.min_window_seconds * SAMPLERATE):
            return UserAffect.NONE
        try:
            self.load()
            valence, arousal = self._predict(audio)
        except SerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad window is not a dead turn
            log.warning("Lane B inference failed on one window: %s", exc)
            return UserAffect.NONE

        self._last = (valence, arousal)
        arousal_z, valence_z = self.calibration.to_z(arousal, valence, self.cfg)
        if arousal_z is None:
            # Uncalibrated, so the absolute number means nothing about
            # HIM. Saying nothing beats reporting an actor-scale value.
            return UserAffect.NONE

        n = self.calibration.n
        if n < self.cfg.affect.warmup_utterances:
            return UserAffect.NONE
        maturity = min(1.0, (n / max(1, self.cfg.affect.baseline_window)) ** 0.5)
        return UserAffect(
            arousal_z=float(arousal_z),
            valence_z=float(valence_z or 0.0),
            confidence=float(max(0.0, min(1.0, 0.4 + 0.6 * maturity))),
            baseline_n=n,
        )

    def commit_utterance(self) -> bool:
        """End of turn: fold this utterance's prediction into the
        calibration. Once per utterance, never per window — the same rule
        as the prosody baseline, for the same reason.
        """
        if self._last is None:
            return False
        valence, arousal = self._last
        self.calibration.observe(arousal, valence)
        self._last = None
        return False
