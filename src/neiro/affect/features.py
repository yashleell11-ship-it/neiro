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
    """Voiced F0 values, and the fraction of frames that were voiced.

    **Pitch by yin, voicing by energy — and that pairing was measured,
    not assumed.** The first version of this used `librosa.pyin` and
    trusted its `voiced_flag`, on the reasoning that a built-in voicing
    decision beats a hand-rolled one. On 60 real CREMA-D clips
    (2026-09-13) that turned out backwards:

        method         usable (voiced_ratio >= 0.15)   mean ratio   ms/clip
        pyin  fl=1024            25/60                    0.135        57
        pyin  fl=2048            24/60                    0.139        51
        pyin  fl=4096            28/60                    0.214        54
        yin   fl=1024            60/60                    0.989         2
        yin   fl=2048            60/60                    0.995         4

    pyin's HMM is far too conservative on short (~2 s) utterances — it
    marked most genuinely voiced speech as unvoiced, at 20x the cost.
    The earlier choice looked right only because it was validated on a
    synthetic tone, which is the easiest possible input for it.

    yin has the opposite flaw: it reports a pitch for silence too, so
    its output alone makes `voiced_ratio` meaninglessly close to 1.0. So
    voicing is decided here, from two cheap and independent signals —
    the frame has enough energy relative to the clip's own peak, and the
    pitch yin found sits inside a plausible human range. That is a
    standard construction, it is ~20x faster than pyin, and unlike pyin
    it works on the audio this project actually sees.
    """
    import librosa

    frame_length = cfg.affect.f0_frame_length
    required = 4 * sr / cfg.affect.f0_min_hz
    if frame_length < required:
        raise ValueError(
            f"f0_frame_length={frame_length} is below librosa's 4*sr/fmin="
            f"{required:.0f} for fmin={cfg.affect.f0_min_hz} Hz at {sr} Hz. "
            "The estimator would return garbage or nothing at all, silently."
        )

    hop = cfg.affect.hop_length
    kwargs = {
        "fmin": cfg.affect.f0_min_hz,
        "fmax": cfg.affect.f0_max_hz,
        "sr": sr,
        "frame_length": frame_length,
        "hop_length": hop,
    }
    if cfg.affect.f0_algorithm == "pyin":
        # Kept as an option — on long, clean, single-speaker audio pyin's
        # voicing is genuinely better. It is not the default because this
        # project's input is short conversational utterances.
        f0, voiced_flag, _ = librosa.pyin(pcm, **kwargs)
        voiced = np.asarray(voiced_flag, dtype=bool)
    else:
        f0 = librosa.yin(pcm, **kwargs)
        voiced = np.ones(np.shape(f0), dtype=bool)

    f0 = np.asarray(f0, dtype=np.float64)

    # Energy gate, computed on the SAME frame grid so the two agree
    # frame-for-frame. Relative to the clip's own peak, because absolute
    # level varies with mic gain and distance by far more than the effect
    # being measured.
    rms = librosa.feature.rms(y=pcm, frame_length=frame_length, hop_length=hop, center=True)[0]
    if rms.size and float(rms.max()) > 0:
        floor = float(rms.max()) * (10.0 ** (-cfg.affect.voiced_db_below_peak / 20.0))
        loud_enough = rms >= floor
    else:
        loud_enough = np.zeros(np.shape(f0), dtype=bool)

    n = min(len(f0), len(loud_enough), len(voiced))
    f0, loud_enough, voiced = f0[:n], loud_enough[:n], voiced[:n]

    voiced &= loud_enough
    voiced &= np.isfinite(f0)
    voiced &= (f0 > cfg.affect.f0_min_hz) & (f0 < cfg.affect.f0_max_hz)

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


def warm(cfg: Neiro | None = None, samplerate: int = 16000) -> float:
    """Run one throwaway extraction so the first real utterance doesn't
    pay the JIT cost. Returns seconds taken.

    Measured on this machine: the first `extract()` call takes ~1210 ms
    and the second ~54 ms — librosa's lazy imports plus numba compiling
    the pyin and onset kernels. That is the same shape of trap Gate G1
    found in ctranslate2 (7.44 s cold → 0.36 s warm), and it has the
    same fix: warm at process start, not on the first thing Yash says in
    the morning.
    """
    import time

    t = np.linspace(0, 1.0, samplerate, endpoint=False)
    fake = (0.4 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)).astype(
        np.float32
    )
    started = time.perf_counter()
    extract(fake, cfg, samplerate)
    return time.perf_counter() - started


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
