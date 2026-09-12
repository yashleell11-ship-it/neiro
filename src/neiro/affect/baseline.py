"""How Yash usually sounds, and how far from it he sounds right now.

This file is the reason the differentiator can work at all. "He sounds
tired" is not a property of an audio clip — it is a comparison against
that specific person's ordinary voice, on that specific microphone. A
classifier trained on actors cannot make that comparison. A rolling
median of his own last fifty utterances can.

**Robust statistics, deliberately.** Median and MAD, never mean and
standard deviation. One shout, one cough, one sentence spoken into a
Bluetooth mic mid-reconnect would drag a mean for the rest of the day;
the median ignores it. `MAD * 1.4826` estimates sigma for normally
distributed data, which is the conversion that makes the output
interpretable as a z-score.

**Keyed by capture device.** The earbuds and the built-in mic have
different frequency responses and, on this laptop, wildly different gain
— the built-in was running at +30 dB. Pooling them would make every
device change look like an emotional event. One baseline per device,
persisted, so she is not emotionally blind for the first minute of every
day.

**Drift detection.** If many consecutive utterances sit absurdly far
from the stored baseline, the baseline is describing someone or
something else — a different speaker, a changed gain, a new headset on
the same node name. That is a reset with a log line, not an emotion.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from neiro.affect.features import FEATURE_NAMES, ProsodyFeatures
from neiro.config import Neiro

log = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".local/state/neiro"
BASELINE_PATH = STATE_DIR / "baseline.json"

# Bump when FEATURE_NAMES changes: old vectors are not comparable to new
# ones, and silently mixing them would corrupt every z-score.
BASELINE_SCHEMA = 1


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _mad(values: list[float], centre: float) -> float:
    return _median([abs(v - centre) for v in values])


@dataclass
class SpeakerBaseline:
    """A rolling window of feature vectors for one capture device."""

    device: str
    window: int = 50
    samples: deque[dict[str, float]] = field(default_factory=deque)
    _drift_run: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.samples, deque) or self.samples.maxlen != self.window:
            self.samples = deque(self.samples, maxlen=self.window)

    @property
    def n(self) -> int:
        return len(self.samples)

    def stats(self) -> dict[str, tuple[float, float]]:
        """`{feature: (median, sigma)}` over the current window."""
        out: dict[str, tuple[float, float]] = {}
        for name in FEATURE_NAMES:
            values = [s[name] for s in self.samples if name in s]
            if not values:
                out[name] = (0.0, 0.0)
                continue
            centre = _median(values)
            out[name] = (centre, _mad(values, centre))
        return out

    def z_scores(self, features: ProsodyFeatures, cfg: Neiro | None = None) -> dict[str, float]:
        """How far this utterance is from his normal, per feature.

        Empty until the window has samples — an unknown baseline produces
        no z-scores at all rather than z=0, because "exactly average" and
        "no idea" must not look the same to the caller.
        """
        cfg = cfg or Neiro()
        if self.n == 0:
            return {}
        out: dict[str, float] = {}
        vector = features.vector()
        for name, (centre, mad) in self.stats().items():
            sigma = mad * cfg.affect.mad_to_sigma
            # A window of near-identical samples gives MAD ~ 0, which
            # would divide a tiny real difference into a huge z. Floor
            # sigma at a fraction of the feature's own scale.
            floor = abs(centre) * cfg.affect.min_sigma_fraction
            sigma = max(sigma, floor)
            if sigma <= 0:
                continue
            z = (vector[name] - centre) / sigma
            out[name] = max(-cfg.affect.max_abs_z, min(cfg.affect.max_abs_z, z))
        return out

    def observe(self, features: ProsodyFeatures, cfg: Neiro | None = None) -> bool:
        """Add one completed utterance. Returns True if drift reset it.

        Called once per *utterance*, never per analysis window — the
        baseline is "how he usually sounds when he says something", and
        counting a 3-second window six times would make one long sentence
        dominate it.
        """
        cfg = cfg or Neiro()
        reset = False
        if self.n >= cfg.affect.warmup_utterances:
            zs = self.z_scores(features, cfg)
            extreme = zs and all(abs(z) >= cfg.affect.drift_z_threshold for z in zs.values())
            self._drift_run = self._drift_run + 1 if extreme else 0
            if self._drift_run >= cfg.affect.drift_consecutive:
                log.warning(
                    "Prosody baseline for %s reset: %d consecutive utterances at "
                    "|z| >= %.1f on every feature. That is a different voice, a "
                    "changed mic gain, or a new headset on the same node name — "
                    "not an emotion.",
                    self.device,
                    self._drift_run,
                    cfg.affect.drift_z_threshold,
                )
                self.samples.clear()
                self._drift_run = 0
                reset = True
        self.samples.append(features.vector())
        return reset

    def to_json(self) -> dict:
        return {
            "device": self.device,
            "window": self.window,
            "samples": list(self.samples),
        }

    @classmethod
    def from_json(cls, data: dict) -> SpeakerBaseline:
        window = int(data.get("window", 50))
        baseline = cls(device=data["device"], window=window)
        baseline.samples = deque(data.get("samples", []), maxlen=window)
        return baseline


@dataclass
class BaselineStore:
    """Every device's baseline, loaded from and saved to one JSON file."""

    path: Path = BASELINE_PATH
    baselines: dict[str, SpeakerBaseline] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None, cfg: Neiro | None = None) -> BaselineStore:
        """Read from disk. A missing, unreadable, or stale-schema file is
        an empty store, never an exception — the worst case is a warm-up
        period, and refusing to start over a corrupt cache would be a far
        worse failure than recalculating it.
        """
        path = path or BASELINE_PATH
        store = cls(path=path)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return store
        if data.get("schema") != BASELINE_SCHEMA:
            log.info(
                "Prosody baseline at %s is schema %s, this build reads %s — starting fresh.",
                path,
                data.get("schema"),
                BASELINE_SCHEMA,
            )
            return store
        for device, payload in (data.get("devices") or {}).items():
            try:
                store.baselines[device] = SpeakerBaseline.from_json(payload)
            except (KeyError, TypeError):
                continue
        return store

    def for_device(self, device: str, cfg: Neiro | None = None) -> SpeakerBaseline:
        cfg = cfg or Neiro()
        if device not in self.baselines:
            self.baselines[device] = SpeakerBaseline(
                device=device, window=cfg.affect.baseline_window
            )
        return self.baselines[device]

    def save(self) -> None:
        """Write atomically. A half-written baseline read at next start
        would be silently wrong rather than loudly broken.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": BASELINE_SCHEMA,
            "features": list(FEATURE_NAMES),
            "devices": {name: b.to_json() for name, b in self.baselines.items()},
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.path)
