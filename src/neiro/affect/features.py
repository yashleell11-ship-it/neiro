"""Raw prosody measurements from one window of speech.

Seven numbers, chosen because each is a well-established correlate of
vocal arousal and each can be measured without a transcript, a model, or
a GPU:

    f0_median_hz      how high he is pitching        — rises with arousal
    f0_iqr_hz         how much the pitch moves       — flat voice vs animated
    rms_mean          how loud, overall              — rises with arousal
    rms_p95           how loud at the peaks          — separates "loud" from "shouty"
    voiced_ratio      how much of the window is voice
    pause_ratio       how much of it is silence      — falls with arousal
    onset_rate_hz     energy peaks per second        — a transcript-free pace proxy

None of these is an emotion. They are the inputs to `baseline.py`, which
turns them into "compared to how he usually sounds" — the only form in
which they mean anything.

**On `None`.** Any window too short, too quiet, or too unvoiced returns
`None` rather than zeros. A zero is a measurement; the absence of one is
not, and the difference matters because the annotation is *omitted*
below the confidence floor rather than softened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from neiro.config import Neiro

# The feature names, in a fixed order. Everything downstream (baseline
# vectors, the persisted JSON, the z-score dict) keys off this, so
# adding a feature is one edit here plus a schema bump in baseline.py.
FEATURE_NAMES: tuple[str, ...] = (
    "f0_median_hz",
    "f0_iqr_hz",
    "rms_mean",
    "rms_p95",
    "voiced_ratio",
    "pause_ratio",
    "onset_rate_hz",
)


@dataclass(frozen=True)
class ProsodyFeatures:
    f0_median_hz: float
    f0_iqr_hz: float
    rms_mean: float
    rms_p95: float
    voiced_ratio: float
    pause_ratio: float
    onset_rate_hz: float
    duration_s: float

    def vector(self) -> dict[str, float]:
        """Just the comparable features — `duration_s` is context, not a
        thing to score against a baseline.
        """
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _rms(pcm: np.ndarray, cfg: Neiro) -> np.ndarray:
    """Frame-wise RMS, computed directly rather than via librosa so this
    module has one less import on the hot path.
    """
    frame, hop = cfg.affect.frame_length, cfg.affect.hop_length
    if len(pcm) < frame:
        return np.array([float(np.sqrt(np.mean(pcm**2)))]) if len(pcm) else np.array([])
    n_frames = 1 + (len(pcm) - frame) // hop
    strides = (pcm.strides[0] * hop, pcm.strides[0])
    frames = np.lib.stride_tricks.as_strided(pcm, shape=(n_frames, frame), strides=strides)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def _pause_ratio(rms: np.ndarray, cfg: Neiro) -> float:
    """Fraction of frames far enough below the window's own peak to be
    silence.

    Relative to the peak, not to an absolute dBFS threshold: mic gain,
    distance from the mic, and the earbuds-vs-speakers profile all shift
    absolute level by more than the effect being measured.
    """
    if rms.size == 0:
        return 0.0
    peak = float(rms.max())
    if peak <= 0:
        return 1.0
    floor = peak * (10.0 ** (-cfg.affect.silence_db_below_peak / 20.0))
    return float(np.mean(rms < floor))


def _f0(pcm: np.ndarray, sr: int, cfg: Neiro) -> tuple[np.ndarray, float]:
    """Voiced F0 values, and the fraction of frames that were voiced."""
    import librosa

    kwargs = {
        "fmin": cfg.affect.f0_min_hz,
        "fmax": cfg.affect.f0_max_hz,
        "sr": sr,
        "frame_length": cfg.affect.frame_length,
        "hop_length": cfg.affect.hop_length,
    }
    if cfg.affect.f0_algorithm == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(pcm, **kwargs)
        voiced = np.asarray(voiced_flag, dtype=bool) & (
            np.asarray(voiced_prob) >= cfg.affect.voiced_prob_floor
        )
    else:
        # yin guesses a pitch for silence too, so voicing has to be
        # inferred from the value landing inside the plausible band.
        f0 = librosa.yin(pcm, **kwargs)
        voiced = np.isfinite(f0) & (f0 > cfg.affect.f0_min_hz) & (f0 < cfg.affect.f0_max_hz)

    f0 = np.asarray(f0, dtype=np.float64)
    voiced &= np.isfinite(f0)
    ratio = float(np.mean(voiced)) if voiced.size else 0.0
    return f0[voiced], ratio


def _onset_rate_hz(pcm: np.ndarray, sr: int, cfg: Neiro) -> float:
    """Energy onsets per second — syllable rate without a transcript.

    Not syllables: an onset detector fires on any sharp energy rise, so
    this over-counts on plosives and under-counts on connected speech.
    It is a *relative* pace measure, which is all a z-score against his
    own baseline needs it to be.
    """
    import librosa

    if len(pcm) < cfg.affect.frame_length:
        return 0.0
    duration = len(pcm) / sr
    strength = librosa.onset.onset_strength(y=pcm, sr=sr, hop_length=cfg.affect.hop_length)
    onsets = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sr, hop_length=cfg.affect.hop_length, units="frames"
    )
    return float(len(onsets) / duration) if duration > 0 else 0.0


def extract(
    pcm_16k: np.ndarray, cfg: Neiro | None = None, samplerate: int = 16000
) -> ProsodyFeatures | None:
    """Measure one window. `None` when the window cannot honestly be
    measured — too short, silent, or almost entirely unvoiced.
    """
    cfg = cfg or Neiro()
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    duration = len(pcm) / samplerate
    if duration < cfg.affect.min_window_seconds:
        return None
    if not np.any(np.isfinite(pcm)) or float(np.max(np.abs(pcm))) <= 0.0:
        return None

    rms = _rms(pcm, cfg)
    f0_voiced, voiced_ratio = _f0(pcm, samplerate, cfg)
    if voiced_ratio < cfg.affect.min_voiced_ratio or f0_voiced.size < 2:
        return None

    q75, q25 = np.percentile(f0_voiced, [75, 25])
    return ProsodyFeatures(
        f0_median_hz=float(np.median(f0_voiced)),
        f0_iqr_hz=float(q75 - q25),
        rms_mean=float(np.mean(rms)) if rms.size else 0.0,
        rms_p95=float(np.percentile(rms, 95)) if rms.size else 0.0,
        voiced_ratio=voiced_ratio,
        pause_ratio=_pause_ratio(rms, cfg),
        onset_rate_hz=_onset_rate_hz(pcm, samplerate, cfg),
        duration_s=duration,
    )
