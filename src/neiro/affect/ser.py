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

**The window is fitted to the checkpoint's clip length, by the recipe's
own function.** Training pads every example to `--seconds` with trailing
zeros or centre-crops it, and w2v-bert's feature extractor normalises
each mel bin over that whole span before the head mean-pools it. This
used to hand the model the raw 3 s rolling window instead — all speech,
no padding — an input composition it had not seen in a single training
example. The length comes from the checkpoint, not from config, because
it is not a tunable: the weights were fitted at exactly that length.
"""

from __future__ import annotations

import logging
import sys
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


def _recipe():
    """The training recipe, imported lazily because it pulls in torch.

    The runtime takes two things from it and re-declares neither: the
    `SerRegressor` architecture and the `fit_clip` length convention. A
    copy of either could drift from what the checkpoint was trained
    with, and nothing in the checkpoint would say so.
    """
    recipes = str(REPO_ROOT / "training" / "recipes")
    if recipes not in sys.path:
        sys.path.insert(0, recipes)
    import ser_train

    return ser_train


def clip_seconds_from(state: dict) -> float:
    """The clip length the checkpoint was trained at, in seconds.

    From the checkpoint and not from config, on purpose: it is not a
    tunable. The weights were fitted to inputs of exactly this length,
    so a runtime value that could disagree with it would be the very bug
    this exists to prevent. A checkpoint that does not record it cannot
    be served honestly, and says so at load time rather than guessing.
    """
    seconds = (state.get("args") or {}).get("seconds")
    if seconds is None:
        raise SerUnavailable(
            "The checkpoint does not record the clip length it was trained at "
            "(`args.seconds`), so the runtime cannot fit the window the way training "
            "did. Retrain with training/recipes/ser_train.py, which saves it."
        )
    return float(seconds)


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
    _clip_seconds: float | None = None  # the checkpoint's; set by load()
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

        recipe = _recipe()
        from transformers import AutoFeatureExtractor

        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self._clip_seconds = clip_seconds_from(state)
        unfreeze = (state.get("args") or {}).get("unfreeze", 4)
        model = recipe.SerRegressor(self.encoder, unfreeze_layers=unfreeze)
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
        """Raw circumplex prediction: `(valence, arousal)` in [-1, 1].

        The window is first fitted to the checkpoint's clip length by the
        recipe's `fit_clip` — trailing zeros for a short one, centre crop
        for a long one — so a 3 s rolling window reaches the feature
        extractor as the same audio-plus-silence every under-length
        training clip did.
        """
        import torch

        if self._clip_seconds is None:
            raise SerUnavailable("Lane B was asked to predict before load() set its clip length")
        audio = _recipe().fit_clip(pcm, int(self._clip_seconds * SAMPLERATE))
        features = self._processor(
            [audio],
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

    def discard_utterance(self) -> None:
        """Drop the staged prediction without calibrating on it."""
        self._last = None
