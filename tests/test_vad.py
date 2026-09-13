"""Silero VAD, the speech gate, and barge-in.

Silero fails quietly in three ways — wrong frame size, dropped RNN
state, un-reset state between utterances — and all three produce
plausible-looking numbers rather than errors. In the component that
decides when Yash has finished talking, that is the worst possible
failure mode, so each one has a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neiro.audio.vad import BargeInDetector, SileroVad, SpeechGate, VadFrameSizeError
from neiro.config import Neiro

MODEL = Path(__file__).resolve().parents[1] / "models" / "silero-vad" / "onnx" / "model.onnx"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="silero-vad not downloaded")


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad()


def speech_like(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    harmonics = 0.5 * np.sin(2 * np.pi * 140 * t) + 0.25 * np.sin(2 * np.pi * 280 * t)
    return (harmonics * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@needs_model
class TestSileroVad:
    def test_silence_is_not_speech(self, vad: SileroVad) -> None:
        vad.reset()
        probs = [vad.probability(np.zeros(vad.frame_samples, dtype=np.float32)) for _ in range(20)]
        assert max(probs) < 0.2

    def test_a_wrong_frame_size_raises_instead_of_guessing(self, vad: SileroVad) -> None:
        # Silero v5 accepts ONLY 512 samples at 16 kHz and returns
        # plausible nonsense for anything else. Loud beats subtle.
        for bad in (256, 1024, 480, 0):
            with pytest.raises(VadFrameSizeError):
                vad.probability(np.zeros(bad, dtype=np.float32))

    def test_the_required_frame_is_32ms_at_16k(self) -> None:
        assert Neiro().vad.frame_samples == 512

    def test_state_is_carried_between_frames(self, vad: SileroVad) -> None:
        # It is a stateful RNN. Dropping the state makes every frame look
        # like the start of speech, which reads as "works, but jumpy"
        # rather than as a bug.
        audio = speech_like(1.0)
        n = vad.frame_samples
        frames = [audio[i : i + n] for i in range(0, len(audio) - n, n)]

        vad.reset()
        carried = [vad.probability(f) for f in frames]

        stateless = []
        for f in frames:
            vad.reset()
            stateless.append(vad.probability(f))

        assert carried != stateless

    def test_reset_returns_it_to_a_clean_start(self, vad: SileroVad) -> None:
        audio = speech_like(0.5)
        n = vad.frame_samples
        vad.reset()
        first = vad.probability(audio[:n])
        for i in range(1, 10):
            vad.probability(audio[i * n : (i + 1) * n])
        vad.reset()
        assert vad.probability(audio[:n]) == pytest.approx(first, abs=1e-6)

    def test_warm_runs_and_leaves_no_residue(self, vad: SileroVad) -> None:
        # Same trap as ctranslate2 and librosa: the first call into a
        # runtime costs far more than the rest.
        assert vad.warm() > 0
        assert vad.probability(np.zeros(vad.frame_samples, dtype=np.float32)) < 0.2

    def test_a_missing_model_says_how_to_fix_it(self) -> None:
        with pytest.raises(FileNotFoundError, match="fetch-models"):
            SileroVad(model_path=Path("/nonexistent/model.onnx"))


class TestSpeechGate:
    """Pure logic — no model needed."""

    def test_one_loud_frame_does_not_start_an_utterance(self) -> None:
        g = SpeechGate()
        assert g.update(0.99) is None
        assert not g.speaking

    def test_sustained_speech_starts_it(self) -> None:
        g = SpeechGate()
        events = [g.update(0.99) for _ in range(6)]
        assert "started" in events
        assert g.speaking

    def test_one_quiet_frame_does_not_end_it(self) -> None:
        # Cutting someone off mid-sentence is much worse than a little
        # trailing silence.
        g = SpeechGate()
        for _ in range(6):
            g.update(0.99)
        assert g.update(0.0) is None
        assert g.speaking

    def test_sustained_silence_ends_it(self) -> None:
        g = SpeechGate()
        for _ in range(6):
            g.update(0.99)
        events = [g.update(0.0) for _ in range(10)]
        assert "stopped" in events
        assert not g.speaking

    def test_leaving_is_slower_than_entering(self) -> None:
        # The same asymmetry as the expression blender, for the same
        # reason: the expensive mistake is the abrupt one.
        cfg = Neiro()
        assert cfg.vad.min_silence_ms > cfg.vad.min_speech_ms

    def test_a_pause_inside_speech_does_not_split_the_turn(self) -> None:
        # "I want to go to... uh... the library" must stay one turn.
        g = SpeechGate()
        for _ in range(6):
            g.update(0.99)
        events = [g.update(0.0) for _ in range(4)]  # ~128 ms hesitation
        events += [g.update(0.99) for _ in range(4)]
        assert "stopped" not in events
        assert g.speaking

    def test_reset_clears_everything(self) -> None:
        g = SpeechGate()
        for _ in range(6):
            g.update(0.99)
        g.reset()
        assert not g.speaking
        assert g.update(0.99) is None


class TestBargeIn:
    def test_her_own_first_syllable_cannot_interrupt_her(self) -> None:
        # The dead zone exists because her voice reaches the mic before
        # any echo canceller has adapted.
        b = BargeInDetector()
        b.start_speaking()
        assert not b.armed
        assert not any(b.update(1.0) for _ in range(5))

    def test_it_arms_after_the_dead_zone(self) -> None:
        b = BargeInDetector()
        b.start_speaking()
        for _ in range(8):
            b.update(0.0)
        assert b.armed

    def test_sustained_loud_speech_after_arming_trips_it(self) -> None:
        b = BargeInDetector()
        b.start_speaking()
        for _ in range(8):
            b.update(0.0)
        assert any(b.update(0.99) for _ in range(12))

    def test_it_is_stricter_than_the_ordinary_gate(self) -> None:
        # A false positive cuts her off mid-sentence; a false negative
        # just means he repeats himself. The costs are not symmetric.
        cfg = Neiro()
        assert cfg.vad.bargein_probability > cfg.vad.threshold
        assert cfg.vad.bargein_speech_ms > cfg.vad.min_speech_ms

    def test_a_brief_noise_does_not_trip_it(self) -> None:
        b = BargeInDetector()
        b.start_speaking()
        for _ in range(8):
            b.update(0.0)
        assert not any([b.update(0.99), b.update(0.99), b.update(0.0)])

    def test_probability_just_under_the_bar_never_trips_it(self) -> None:
        b = BargeInDetector()
        b.start_speaking()
        for _ in range(8):
            b.update(0.0)
        under = Neiro().vad.bargein_probability - 0.01
        assert not any(b.update(under) for _ in range(40))

    def test_starting_a_new_reply_re_arms_the_dead_zone(self) -> None:
        b = BargeInDetector()
        b.start_speaking()
        for _ in range(20):
            b.update(0.0)
        assert b.armed
        b.start_speaking()
        assert not b.armed


@needs_model
class TestAgainstRealSpeech:
    def test_it_separates_real_speech_from_silence(self, vad: SileroVad) -> None:
        corpus = Path(__file__).resolve().parents[1] / "data/datasets/crema-d/extracted"
        clips = sorted(corpus.rglob("1001_*.wav"))[:3] if corpus.is_dir() else []
        if not clips:
            pytest.skip("CREMA-D not extracted")
        import soundfile as sf

        n = vad.frame_samples
        for clip in clips:
            audio, _ = sf.read(clip, dtype="float32")
            vad.reset()
            probs = [vad.probability(audio[i : i + n]) for i in range(0, len(audio) - n, n)]
            assert max(probs) > 0.8, f"{clip.name}: no frame looked like speech"

        vad.reset()
        silence = [vad.probability(np.zeros(n, dtype=np.float32)) for _ in range(30)]
        assert max(silence) < 0.2
