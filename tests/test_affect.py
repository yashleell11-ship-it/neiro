"""Tests for the differentiator: prosody → baseline → UserAffect.

The rules worth pinning here are the ones about *refusing to speak*. An
emotion system that always has an opinion is worse than none, because
the character acts on it. So: no baseline means no reading, a short
warm-up means zero confidence (not low), and inside the dead-band the
annotation is omitted entirely rather than softened to "he sounds
normal".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from neiro.affect import features as feat
from neiro.affect.baseline import BASELINE_SCHEMA, BaselineStore, SpeakerBaseline
from neiro.affect.labels import CIRCUMPLEX
from neiro.affect.null import NullAffectProvider
from neiro.affect.prosody import (
    AROUSAL_WEIGHTS,
    ProsodyAffectProvider,
    band_for,
    describe,
)
from neiro.config import Neiro
from neiro.state import Locality, Tier, UserAffect

SR = 16000


def speech_like(f0: float, amp: float, rate: float = 3.0, dur: float = 3.0) -> np.ndarray:
    """A voiced, harmonically rich, amplitude-modulated tone.

    Not speech — but it has a well-defined pitch, energy and syllable-ish
    rate, which is exactly the three things the features measure. Real
    speech is Yash's job (gate G3b); this pins the arithmetic.
    """
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    harmonics = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    return (harmonics * (0.6 + 0.4 * np.sin(2 * np.pi * rate * t)) * amp).astype(np.float32)


CALM = speech_like(f0=130, amp=0.25, rate=2.5)
EXCITED = speech_like(f0=190, amp=0.70, rate=5.0)


class TestFeatures:
    def test_measures_a_voiced_window(self) -> None:
        f = feat.extract(CALM)
        assert f is not None
        assert 100 < f.f0_median_hz < 170
        assert f.rms_mean > 0
        assert 0 < f.voiced_ratio <= 1
        assert f.duration_s == pytest.approx(3.0, abs=0.05)

    def test_excited_reads_louder_higher_and_faster(self) -> None:
        # The whole premise in one assertion. If this ever fails, the
        # features are not measuring what the design says they measure.
        calm, excited = feat.extract(CALM), feat.extract(EXCITED)
        assert calm is not None and excited is not None
        assert excited.f0_median_hz > calm.f0_median_hz
        assert excited.rms_mean > calm.rms_mean
        assert excited.onset_rate_hz > calm.onset_rate_hz

    @pytest.mark.parametrize(
        ("name", "pcm"),
        [
            ("digital silence", np.zeros(SR * 3, dtype=np.float32)),
            ("too short", speech_like(150, 0.5, dur=0.3)),
            ("empty", np.zeros(0, dtype=np.float32)),
        ],
    )
    def test_unmeasurable_windows_return_none_not_zeros(self, name: str, pcm: np.ndarray) -> None:
        # None and 0.0 must not be confusable: one is "no reading", the
        # other is a measurement. The whole omit-don't-soften rule
        # downstream depends on the difference.
        assert feat.extract(pcm) is None, name

    def test_broadband_noise_is_measured_and_that_is_a_known_limitation(self) -> None:
        """Pure white noise gets features rather than None, deliberately.

        Every discriminator measured on 2026-09-13 that rejects noise
        also rejects real speech:

          - F0 contour continuity: white noise 0.765 vs real speech
            0.800 (min 0.661). They overlap; no threshold separates them.
          - Spectral flatness: white noise 0.56 vs real speech 0.0096, a
            clean 20x separation — BUT the same clip plus mild noise at
            roughly 30 dB SNR reads 0.51. A gate strict enough to reject
            noise would reject Yash speaking in a hostel room, which is
            the entire deployment environment.

        So it is handled upstream instead: audio reaches the affect path
        only behind push-to-talk (Stage 0) and Silero VAD (Stage 2).
        Broadband noise is not a realistic input here, and pretending to
        filter it would cost real utterances.
        """
        noise = (np.random.default_rng(0).standard_normal(SR * 3) * 0.1).astype(np.float32)
        assert feat.extract(noise) is not None

    def test_quiet_frames_inside_a_window_are_not_counted_as_voice(self) -> None:
        # What the energy gate does buy: room tone between words is
        # excluded, so voiced_ratio means something.
        rng = np.random.default_rng(1)
        loud = speech_like(f0=140, amp=0.6, dur=1.5)
        quiet = (rng.standard_normal(int(SR * 1.5)) * 0.001).astype(np.float32)
        f = feat.extract(np.concatenate([loud, quiet]))
        assert f is not None
        assert f.voiced_ratio < 0.75, "the silent half must not count as voiced"

    def test_nan_input_does_not_crash(self) -> None:
        pcm = np.full(SR * 3, np.nan, dtype=np.float32)
        assert feat.extract(pcm) is None

    def test_vector_excludes_duration(self) -> None:
        # duration_s is context for the confidence calculation, not a
        # thing to score against a baseline — a long sentence is not an
        # emotional state.
        f = feat.extract(CALM)
        assert f is not None
        assert set(f.vector()) == set(feat.FEATURE_NAMES)
        assert "duration_s" not in f.vector()
        assert "duration_s" in f.as_dict()


class TestBaseline:
    def test_no_samples_means_no_z_scores(self) -> None:
        # Not z=0. "Exactly average" and "no idea" must never look alike.
        b = SpeakerBaseline(device="earbuds")
        f = feat.extract(CALM)
        assert f is not None
        assert b.z_scores(f) == {}

    def test_z_is_zero_against_a_baseline_of_itself(self) -> None:
        b = SpeakerBaseline(device="earbuds")
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(10):
            b.observe(f)
        for name, z in b.z_scores(f).items():
            assert abs(z) < 0.5, name

    def test_excited_scores_positive_against_a_calm_baseline(self) -> None:
        b = SpeakerBaseline(device="earbuds")
        rng = np.random.default_rng(1)
        for _ in range(20):
            jitter = speech_like(
                f0=130 + rng.normal(0, 4), amp=0.25 + rng.normal(0, 0.02), rate=2.5
            )
            f = feat.extract(jitter)
            if f:
                b.observe(f)
        excited = feat.extract(EXCITED)
        assert excited is not None
        zs = b.z_scores(excited)
        assert zs["rms_mean"] > 1.0
        assert zs["f0_median_hz"] > 1.0

    def test_window_is_bounded(self) -> None:
        b = SpeakerBaseline(device="x", window=5)
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(20):
            b.observe(f)
        assert b.n == 5

    def test_z_is_clamped(self) -> None:
        cfg = Neiro()
        b = SpeakerBaseline(device="x")
        quiet = feat.extract(speech_like(130, 0.05, 2.0))
        loud = feat.extract(speech_like(380, 0.95, 9.0))
        assert quiet is not None and loud is not None
        for _ in range(10):
            b.observe(quiet)
        for z in b.z_scores(loud).values():
            assert abs(z) <= cfg.affect.max_abs_z

    def test_one_outlier_does_not_move_the_baseline(self) -> None:
        # The reason for median/MAD over mean/std: a single shout must
        # not redefine his normal for the rest of the day.
        b = SpeakerBaseline(device="x")
        calm_f = feat.extract(CALM)
        shout = feat.extract(speech_like(320, 0.95, 8.0))
        assert calm_f is not None and shout is not None
        for _ in range(20):
            b.observe(calm_f)
        before = b.stats()["rms_mean"][0]
        b.observe(shout)
        after = b.stats()["rms_mean"][0]
        assert abs(after - before) / before < 0.1


class TestDrift:
    def test_a_sustained_new_voice_resets_the_baseline(self) -> None:
        cfg = Neiro()
        b = SpeakerBaseline(device="earbuds")
        calm_f = feat.extract(CALM)
        assert calm_f is not None
        for _ in range(cfg.affect.baseline_window):
            b.observe(calm_f, cfg)
        other = feat.extract(speech_like(380, 0.95, 9.0))
        assert other is not None
        resets = [b.observe(other, cfg) for _ in range(cfg.affect.drift_consecutive + 2)]
        assert any(resets), "a sustained, extreme shift should reset, not be read as emotion"

    def test_a_realistic_speaker_change_resets_even_though_only_pitch_moves(self) -> None:
        # The case the original rule missed. A different person mostly
        # shifts PITCH; pausing and voiced ratio stay similar. Requiring
        # every feature to be extreme meant drift never fired at all.
        cfg = Neiro()
        b = SpeakerBaseline(device="earbuds")
        mine = feat.extract(speech_like(f0=120, amp=0.30, rate=3.0))
        theirs = feat.extract(speech_like(f0=250, amp=0.32, rate=3.0))  # only pitch differs
        assert mine is not None and theirs is not None
        for _ in range(cfg.affect.baseline_window):
            b.observe(mine, cfg)
        resets = [b.observe(theirs, cfg) for _ in range(cfg.affect.drift_consecutive + 2)]
        assert any(resets)

    def test_one_loud_utterance_does_not_reset(self) -> None:
        cfg = Neiro()
        b = SpeakerBaseline(device="earbuds")
        calm_f = feat.extract(CALM)
        loud = feat.extract(speech_like(380, 0.95, 9.0))
        assert calm_f is not None and loud is not None
        for _ in range(cfg.affect.baseline_window):
            b.observe(calm_f, cfg)
        assert b.observe(loud, cfg) is False
        assert b.n > 1


class TestPersistence:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "baseline.json"
        store = BaselineStore(path=path)
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(6):
            store.for_device("earbuds").observe(f)
        store.save()

        reloaded = BaselineStore.load(path)
        assert reloaded.for_device("earbuds").n == 6

    def test_devices_are_kept_apart(self, tmp_path) -> None:
        # The built-in mic ran at +30 dB on this laptop. Pooling devices
        # would make every switch look like an emotional event.
        store = BaselineStore(path=tmp_path / "b.json")
        calm_f, loud = feat.extract(CALM), feat.extract(EXCITED)
        assert calm_f is not None and loud is not None
        for _ in range(10):
            store.for_device("earbuds").observe(calm_f)
            store.for_device("speakers").observe(loud)
        assert (
            store.for_device("earbuds").stats()["rms_mean"][0]
            < store.for_device("speakers").stats()["rms_mean"][0]
        )

    def test_a_missing_file_is_an_empty_store_not_an_error(self, tmp_path) -> None:
        assert BaselineStore.load(tmp_path / "nope.json").baselines == {}

    def test_corrupt_json_is_an_empty_store_not_an_error(self, tmp_path) -> None:
        # Refusing to start over a corrupt cache would be a far worse
        # failure than a warm-up period.
        path = tmp_path / "b.json"
        path.write_text("{not json")
        assert BaselineStore.load(path).baselines == {}

    def test_a_stale_schema_is_discarded(self, tmp_path) -> None:
        # Old feature vectors are not comparable to new ones; mixing them
        # would corrupt every z-score silently. The fixture is what THIS
        # build writes with only the schema number changed, and the
        # positive control proves it loads as written — so the schema
        # guard is the only thing that can empty the store. The earlier
        # fixture had an empty device payload, which from_json rejected
        # on its own: the test passed with the guard deleted.
        path = tmp_path / "b.json"
        store = BaselineStore(path=path)
        f = feat.extract(CALM)
        assert f is not None
        store.for_device("x").observe(f)
        store.save()
        assert BaselineStore.load(path).baselines["x"].n == 1  # positive control

        stale = json.loads(path.read_text())
        stale["schema"] = BASELINE_SCHEMA + 99
        path.write_text(json.dumps(stale))
        assert BaselineStore.load(path).baselines == {}

    def test_save_leaves_no_temp_file_behind(self, tmp_path) -> None:
        # A save that copied instead of renaming would leak the .tmp.
        # This catches that and nothing more — atomicity is the next test.
        path = tmp_path / "b.json"
        store = BaselineStore(path=path)
        store.for_device("x")
        store.save()
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()
        assert json.loads(path.read_text())["schema"] == BASELINE_SCHEMA

    @pytest.mark.parametrize("failing_step", ["fsync", "replace"])
    def test_a_crash_mid_save_leaves_the_previous_baseline_intact(
        self, tmp_path, monkeypatch, failing_step: str
    ) -> None:
        # "Atomic" means a crash at ANY point during save leaves the old
        # file whole and readable — never truncated, which load() would
        # swallow as corrupt and silently restart from zero. The real
        # file must not change until the new bytes are complete and on
        # disk. The test this replaces only checked that no .tmp was left
        # behind, which a plain in-place write satisfies just as well.
        path = tmp_path / "b.json"
        store = BaselineStore(path=path)
        store.for_device("x")
        store.save()
        good = path.read_text()

        def crash(*args, **kwargs):
            raise OSError(f"simulated crash at {failing_step}")

        monkeypatch.setattr(os, failing_step, crash)
        store.for_device("y")
        with pytest.raises(OSError):
            store.save()
        assert path.read_text() == good
        assert BaselineStore.load(path).baselines.keys() == {"x"}


class TestBands:
    def test_dead_band(self) -> None:
        assert band_for(0.0, 0.5) == "usual"
        assert band_for(0.49, 0.5) == "usual"
        assert band_for(0.5, 0.5) == "higher"
        assert band_for(-0.5, 0.5) == "lower"

    def test_arousal_weights_point_the_established_way(self) -> None:
        # Louder, higher, faster = more activated; more pausing = less.
        # If a sign here ever flips, the whole thing reads backwards.
        assert AROUSAL_WEIGHTS["rms_mean"] > 0
        assert AROUSAL_WEIGHTS["f0_median_hz"] > 0
        assert AROUSAL_WEIGHTS["onset_rate_hz"] > 0
        assert AROUSAL_WEIGHTS["pause_ratio"] < 0


class TestDescribe:
    def test_low_confidence_is_omitted_entirely(self) -> None:
        # Omitted, not softened — see prompts/neiro.v1.md.
        assert describe(UserAffect(arousal_z=2.0, confidence=0.1)) is None

    def test_inside_the_dead_band_is_omitted_entirely(self) -> None:
        # "He sounds normal" every turn is noise she will act on.
        assert describe(UserAffect(arousal_z=0.1, confidence=0.9)) is None

    def test_high_arousal_is_described_in_words_not_numbers(self) -> None:
        text = describe(UserAffect(arousal_z=2.0, confidence=0.9))
        assert text is not None
        assert "energy" in text
        assert "σ" not in text and "z" not in text.split()

    def test_low_arousal_reads_as_flat(self) -> None:
        text = describe(UserAffect(arousal_z=-2.0, confidence=0.9))
        assert text is not None and "flatter" in text

    def test_confidence_is_stated_so_the_prompt_can_ignore_it(self) -> None:
        assert "tone-confidence" in (describe(UserAffect(arousal_z=2.0, confidence=0.9)) or "")


class TestProvider:
    def _provider(self, tmp_path) -> ProsodyAffectProvider:
        return ProsodyAffectProvider(
            cfg=Neiro(), device="earbuds", store=BaselineStore(path=tmp_path / "b.json")
        )

    def test_is_local_pinned(self) -> None:
        # The mic is wherever Yash is. This must never be tierable — and
        # the rule is checked through `allows()`, which is what a future
        # TierResolver will actually consult, not through the enum name.
        assert ProsodyAffectProvider.locality is Locality.LOCAL_PINNED
        assert ProsodyAffectProvider.locality.allows(Tier.LOCAL)
        assert not ProsodyAffectProvider.locality.allows(Tier.LAN)
        assert not ProsodyAffectProvider.locality.allows(Tier.TUNNEL)

    def test_warm_actually_runs(self, tmp_path) -> None:
        # Regression: `warm()` was referenced by the provider but never
        # existed in features.py, so calling it raised AttributeError at
        # runtime -- and nothing exercised it. An edit had silently
        # no-opped. Any public method the provider offers gets called by
        # a test, or it is not known to work.
        p = self._provider(tmp_path)
        elapsed = p.warm()
        assert elapsed > 0
        assert feat.warm(p.cfg) > 0

    def test_cold_provider_says_nothing(self, tmp_path) -> None:
        p = self._provider(tmp_path)
        assert asyncio.run(p.observe(CALM)) == UserAffect.NONE

    def test_confidence_is_zero_during_warmup_not_merely_low(self, tmp_path) -> None:
        p = self._provider(tmp_path)
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(p.cfg.affect.warmup_utterances - 1):
            p.baseline.observe(f, p.cfg)
        assert asyncio.run(p.observe(EXCITED)).confidence == 0.0

    def test_a_trained_provider_hears_excitement(self, tmp_path) -> None:
        p = self._provider(tmp_path)
        rng = np.random.default_rng(2)
        for _ in range(30):
            f = feat.extract(speech_like(130 + rng.normal(0, 4), 0.25 + rng.normal(0, 0.02), 2.5))
            if f:
                p.baseline.observe(f, p.cfg)
        # Three windows so hysteresis can settle on the same band.
        for _ in range(3):
            affect = asyncio.run(p.observe(EXCITED))
        assert affect.arousal_z > p.cfg.affect.dead_band_z
        assert affect.confidence > p.cfg.affect.confidence_floor
        assert describe(affect, p.cfg) is not None

    def test_hysteresis_needs_agreement_before_the_band_moves(self, tmp_path) -> None:
        # With a history present, one disagreeing window must not move
        # the band. (With NO history there is nothing to disagree with,
        # which is a separate case -- see the short-utterance test.)
        p = self._provider(tmp_path)
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(30):
            p.baseline.observe(f, p.cfg)
        for _ in range(3):
            asyncio.run(p.observe(CALM))  # establish a "usual" history
        assert p._band == "usual"
        asyncio.run(p.observe(EXCITED))
        assert p._band == "usual", "one window must not flip her face"
        asyncio.run(p.observe(EXCITED))
        assert p._band == "higher"

    def test_a_single_window_utterance_is_not_penalised_for_having_no_history(
        self, tmp_path
    ) -> None:
        # A 2-3 second sentence produces ONE analysis window. Requiring
        # hysteresis agreement there meant every short utterance was
        # charged for contradicting an empty deque, so a brief "what?!"
        # could never produce a reading -- exactly the utterances most
        # likely to carry emotion.
        p = self._provider(tmp_path)
        f = feat.extract(CALM)
        assert f is not None
        for _ in range(30):
            p.baseline.observe(f, p.cfg)
        first = asyncio.run(p.observe(EXCITED))
        assert p._band == "higher"
        assert first.confidence > 0

    def test_commit_updates_the_baseline_once_per_utterance(self, tmp_path) -> None:
        # Six analysis windows of one sentence must add ONE sample, or a
        # single long utterance dominates his normal.
        p = self._provider(tmp_path)
        for _ in range(6):
            asyncio.run(p.observe(CALM))
        before = p.baseline.n
        p.commit_utterance()
        assert p.baseline.n == before + 1

    def test_commit_with_nothing_observed_is_a_no_op(self, tmp_path) -> None:
        p = self._provider(tmp_path)
        assert p.commit_utterance() is False
        assert p.baseline.n == 0

    def test_valence_is_never_as_confident_as_arousal(self, tmp_path) -> None:
        # Measured at G3b, not trusted before it.
        p = self._provider(tmp_path)
        affect = UserAffect(arousal_z=2.0, valence_z=1.5, confidence=0.9)
        assert p.valence_confidence(affect) < affect.confidence


class TestNullProvider:
    def test_hears_nothing_and_says_so(self) -> None:
        p = NullAffectProvider()
        assert asyncio.run(p.observe(EXCITED)) == UserAffect.NONE
        assert p.locality is Locality.LOCAL_PINNED

    def test_is_interchangeable_with_the_real_one(self) -> None:
        # The orchestrator must not be able to tell them apart — turning
        # affect on is a config flip, not a code path.
        for name in ("observe", "commit_utterance", "locality"):
            assert hasattr(NullAffectProvider, name), name


class TestLaneBProvider:
    """The trained model at runtime. No checkpoint needed for most of it."""

    def test_it_is_local_pinned(self) -> None:
        # It reads the microphone, and audio is the one thing not worth
        # sending over a link.
        from neiro.affect.ser import SerAffectProvider

        assert SerAffectProvider.locality is Locality.LOCAL_PINNED
        assert not SerAffectProvider.locality.allows(Tier.LAN)

    def test_a_missing_checkpoint_says_how_to_train_one(self, tmp_path) -> None:
        # At construction time, not at the first utterance: a model that
        # cannot load should stop start-up, not fail mid-conversation.
        from neiro.affect.ser import SerAffectProvider, SerUnavailable

        p = SerAffectProvider(checkpoint=tmp_path / "nope.pt")
        with pytest.raises(SerUnavailable, match="ser_train"):
            p.load()

    def test_it_is_interchangeable_with_the_other_providers(self) -> None:
        # Turning Lane B on must be a config flip, not a code path.
        from neiro.affect.ser import SerAffectProvider

        for name in ("observe", "commit_utterance", "warm", "locality"):
            assert hasattr(SerAffectProvider, name), name

    def test_it_is_off_by_default_and_paced_slower_than_lane_a(self) -> None:
        # Measured: ~590 ms per window against Lane A's ~10 ms, on a
        # 750 ms cadence with STT and a browser also running.
        cfg = Neiro()
        assert cfg.affect.lane_b_enabled is False
        assert cfg.affect.lane_b_interval_s > 0.75

    def test_an_uncalibrated_reading_says_nothing(self) -> None:
        # The model predicts absolute circumplex coordinates learned from
        # actors. Until it has been baselined on HIM, that number means
        # nothing about him, and reporting it would be worse than silence.
        import asyncio

        from neiro.affect.ser import SerAffectProvider

        p = SerAffectProvider()
        assert p.calibration.n == 0
        assert asyncio.run(p.observe(np.zeros(100, dtype=np.float32))) == UserAffect.NONE


# --------------------------------------------------------------------------
# Lane B and its recipe: what must mean the same at both ends.
#
# `training/recipes/ser_train.py` and `affect/ser.py` are two ends of one
# model. Pinned here are the things that have to agree across them — the
# definition of "high arousal" the reported AUC is scored against, and the
# clip length a waveform is fitted to before the feature extractor sees
# it. Each once had a private version on one end, and the numbers in
# report.json quietly stopped describing the model that runs.

NEUTRAL = CIRCUMPLEX["neutral"]


@pytest.fixture(scope="module")
def recipe():
    """`ser_train`, imported the way `affect/ser.py` imports it.

    Skipped, not failed, where torch is absent: the recipe needs it at
    import time, and nothing above this line does.
    """
    pytest.importorskip("torch")
    recipes = str(Path(__file__).resolve().parents[1] / "training" / "recipes")
    if recipes not in sys.path:
        sys.path.insert(0, recipes)
    import ser_train

    return ser_train


class TestAucAbove:
    """Gate G3b's number, scored against ONE class definition."""

    # A crema-d-like subset: anger, fear, three happy, one neutral. The
    # predictions rank the happy clips below the neutral one.
    TRUE = np.array([0.8, 0.7, 0.5, 0.5, 0.5, 0.0])
    PRED = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.5])

    def test_the_class_is_the_boundary_not_the_subsets_median(self, recipe) -> None:
        # Above neutral: five positives against one negative, and three
        # of the five rank below it — 2 of 5 pairs ordered correctly.
        assert recipe.auc_above(self.PRED, self.TRUE, NEUTRAL.arousal) == pytest.approx(0.4)
        # The old rule split this subset at its own median, 0.5, which
        # put every `happy` on the LOW side and scored the very same
        # predictions a perfect 1.0. Same model, same clips, and the
        # number depended on what else happened to be in the subset.
        assert recipe.auc_above(self.PRED, self.TRUE, float(np.median(self.TRUE))) == 1.0

    def test_the_same_label_is_the_same_class_in_every_subset(self, recipe) -> None:
        # The real failure: `fear` (0.7) was a positive in the crema-d
        # row of report.json and a negative in the rasa row, because
        # each row was split at its own median. A model that ranks one
        # fear clip above everything else, and is otherwise at chance,
        # must read as better than chance on BOTH subsets.
        fear = 0.7
        crema_like = np.array([fear, 0.8, 0.5, 0.5, 0.0, -0.4])  # median 0.5
        rasa_like = np.array([fear, 0.8, 0.8, 0.7, 0.7, 0.0, -0.4])  # median 0.7
        for subset in (crema_like, rasa_like):
            pred = np.zeros_like(subset)
            pred[0] = 1.0
            assert recipe.auc_above(pred, subset, NEUTRAL.arousal) > 0.5
        # Split at each subset's own median, that identical behaviour
        # scored 0.75 on one corpus and 0.4 — worse than chance — on the
        # other, because the rasa median made `fear` a negative there.
        pred = np.zeros_like(rasa_like)
        pred[0] = 1.0
        assert recipe.auc_above(pred, rasa_like, float(np.median(rasa_like))) == pytest.approx(0.4)

    def test_one_class_is_undefined_not_chance(self, recipe) -> None:
        # The old copy returned 0.5 here, indistinguishable from a model
        # that genuinely cannot tell the classes apart.
        assert recipe.auc_above(np.array([0.1, 0.9]), np.array([0.8, 0.7]), NEUTRAL.arousal) is None

    def test_a_constant_predictor_scores_chance(self, recipe) -> None:
        assert recipe.auc_above(np.full(6, 0.3), self.TRUE, NEUTRAL.arousal) == 0.5


class TestEvaluate:
    def test_every_subset_is_scored_against_the_run_boundary(self, recipe) -> None:
        import torch

        # An identity "model": the features handed in ARE its predictions,
        # so the one-batch loader below fixes both sides exactly.
        true = torch.tensor([[0.0, a] for a in TestAucAbove.TRUE], dtype=torch.float32)
        pred = torch.tensor([[0.0, a] for a in TestAucAbove.PRED], dtype=torch.float32)
        scores = recipe.evaluate(
            torch.nn.Identity(),
            [(pred, true)],
            "cpu",
            torch.float32,
            arousal_boundary=NEUTRAL.arousal,
            valence_boundary=NEUTRAL.valence,
        )
        assert scores["n"] == 6
        assert scores["arousal_auc"] == 0.4
        # Every valence label sits AT the boundary: one class, so the
        # number is absent from the report rather than a fake 0.5.
        assert scores["valence_auc"] is None

    def test_the_boundaries_are_explicit_and_land_in_the_checkpoint(self, recipe) -> None:
        args = recipe.build_parser().parse_args([])
        assert args.arousal_boundary == NEUTRAL.arousal
        assert args.valence_boundary == NEUTRAL.valence
        # `vars(args)` is what main() writes into best.pt and into
        # report.json's `config`, so a reported AUC can always be traced
        # to the class definition it was scored against.
        assert {"arousal_boundary", "valence_boundary", "seconds"} <= set(vars(args))
