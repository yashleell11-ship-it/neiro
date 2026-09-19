"""The wake word, tested without onnxruntime and without a microphone.

Every mistake available in `wake.py` is a SILENT one. Feed the embedder
un-scaled mel, or hand it 512-sample frames because that is what Silero
wants, or let one spoken phrase score on six consecutive hops — none of
those raise. They produce a detector that loads cleanly, reports no
errors, and never wakes up, which is indistinguishable from a bad
threshold and has nothing in common with it as a fix.

So the three ONNX sessions are faked here and the assertions are about
what reaches them: the SHAPE of the window, the SCALING of the mel, and
the number of times a single crossing is allowed to fire.

The pipeline itself — that these shapes and this scaling are the ones
openWakeWord's extractors actually want — is not knowable from fakes and
was not assumed. It was measured end-to-end against a head somebody else
trained (hey_glados, apache-2.0) driven by this project's own Kokoro
voice: 0.841 and 0.623 peak on "Hey GLaDOS" across two voices, 0.001 on
"what is the weather today" and on "Hey Elizabeth". See
docs/DECISIONS.md, 2026-09-20.
"""

from __future__ import annotations

import numpy as np
import pytest

from elizabeth.audio.wake import WakeEvent, WakeWord
from elizabeth.config import Elizabeth, WakeConfig

MEL_FRAMES_PER_HOP = 5  # measured: 1280 samples -> 5 mel frames of 32 bins


class FakeSession:
    """Records what it was called with and returns a fixed shape."""

    def __init__(self, out: np.ndarray, input_name: str = "x") -> None:
        self.out = out
        self.calls: list[dict] = []
        self._name = input_name

    def run(self, _outputs, feed):
        self.calls.append(feed)
        return [self.out]

    def get_inputs(self):
        class _I:
            name = self._name
        return [_I()]


class ScriptedHead(FakeSession):
    """A head whose score is read off a list, one per call."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__(np.zeros((1, 1), dtype=np.float32), input_name="head_in")
        self.scores = list(scores)

    def run(self, _outputs, feed):
        self.calls.append(feed)
        score = self.scores.pop(0) if self.scores else 0.0
        return [np.array([[score]], dtype=np.float32)]


def build(scores: list[float], **over) -> tuple[WakeWord, FakeSession, FakeSession, ScriptedHead]:
    cfg = WakeConfig(**over)
    mel = FakeSession(np.ones((1, 1, MEL_FRAMES_PER_HOP, 32), dtype=np.float32) * 10.0,
                      input_name="input")
    emb = FakeSession(np.ones((1, 1, 1, 96), dtype=np.float32), input_name="input_1")
    head = ScriptedHead(scores)
    return WakeWord(cfg, features=(mel, emb), head=head), mel, emb, head


def feed_hops(w: WakeWord, n: int, *, start: float = 0.0, block: int = 512) -> list[WakeEvent]:
    """`n` hops' worth of audio, delivered in `listen.py`-sized blocks."""
    total = n * 1280
    audio = np.full(total, 0.1, dtype=np.float32)
    events = []
    for i in range(0, total, block):
        ev = w.feed(audio[i:i + block], now=start + i / 16000.0)
        if ev is not None:
            events.append(ev)
    return events


class TestFrameSizing:
    def test_512_sample_blocks_are_rebuffered_into_1280_sample_hops(self) -> None:
        # listen.py and vad.py both work in 512-sample Silero windows. The
        # wake front end takes 1280. Neither side should have to know that,
        # and handing the mel model a 512-sample frame does not raise — it
        # just produces the wrong number of mel frames forever.
        w, mel, _, _ = build([0.0] * 50)
        primed = len(mel.calls)
        feed_hops(w, 4, block=512)
        hops = len(mel.calls) - primed
        assert hops == 4, f"4 hops of audio should be 4 mel calls, got {hops}"
        for feed in mel.calls[primed:]:
            assert feed["input"].shape == (1, 1280)

    def test_a_partial_block_is_held_not_dropped(self) -> None:
        w, mel, _, _ = build([0.0] * 50)
        primed = len(mel.calls)
        w.feed(np.zeros(700, dtype=np.float32), now=0.0)
        assert len(mel.calls) == primed, "700 samples is not a hop yet"
        w.feed(np.zeros(580, dtype=np.float32), now=0.1)
        assert len(mel.calls) == primed + 1, "700 + 580 = 1280 = exactly one hop"

    def test_audio_reaches_the_front_end_in_int16_range(self) -> None:
        # The ONNX graphs were traced on int16-range values; our audio is
        # float [-1, 1] everywhere else in this project. Getting this wrong
        # is silent: the mel is simply 90 dB quieter than training.
        w, mel, _, _ = build([0.0] * 50)
        primed = len(mel.calls)
        w.feed(np.ones(1280, dtype=np.float32), now=0.0)
        sent = mel.calls[primed]["input"]
        assert sent.max() == pytest.approx(32767.0, rel=1e-3)


class TestWindowsAndScaling:
    def test_the_embedder_gets_exactly_76_mel_frames_of_32_bins(self) -> None:
        w, _, emb, _ = build([0.0] * 50)
        feed_hops(w, 2)
        assert emb.calls, "no embedding was ever computed"
        assert emb.calls[-1]["input_1"].shape == (1, 76, 32, 1)

    def test_the_mel_is_scaled_before_the_embedder_sees_it(self) -> None:
        # mel/10 + 2. Skipping it does not error; it feeds the classifier a
        # distribution it never saw, which looks like "never fires".
        w, _, emb, _ = build([0.0] * 50)
        feed_hops(w, 2)
        window = emb.calls[-1]["input_1"]
        # The fake mel emits a constant 10.0, so every scaled value is 3.0.
        assert np.allclose(window, 10.0 / 10.0 + 2.0)

    def test_the_head_gets_exactly_16_embeddings_of_96(self) -> None:
        w, _, _, head = build([0.0] * 50)
        feed_hops(w, 20)
        assert head.calls, "the head was never called"
        assert head.calls[-1]["head_in"].shape == (1, 16, 96)

    def test_nothing_is_scored_until_the_context_is_full(self) -> None:
        # 16 embeddings at one per hop. Scoring earlier means scoring a
        # window half full of priming silence.
        w, _, _, head = build([0.9] * 50)
        feed_hops(w, 15)
        assert head.calls == [], "scored before 16 embeddings existed"
        feed_hops(w, 1, start=10.0)
        assert len(head.calls) == 1


class TestFiring:
    def test_it_fires_when_the_score_crosses(self) -> None:
        # The head is not called until hop 16 -- the first one with a full
        # 16-embedding context -- so the scripted scores are consumed from
        # hop 16 onward, not from hop 1.
        w, _, _, _ = build([0.0, 0.9])
        events = feed_hops(w, 17)
        assert len(events) == 1
        assert events[0].score == pytest.approx(0.9)

    def test_one_phrase_fires_once(self) -> None:
        # A spoken phrase crosses on several consecutive hops. Without the
        # refractory window she answers her own name three times.
        w, _, _, _ = build([0.99] * 40, refractory_ms=2000.0)
        events = feed_hops(w, 30)
        assert len(events) == 1, f"one crossing became {len(events)} events"

    def test_the_refractory_window_expires(self) -> None:
        w, _, _, _ = build([0.99] * 80, refractory_ms=500.0)
        first = feed_hops(w, 20)
        second = feed_hops(w, 20, start=60.0)  # well past the window
        assert len(first) == 1 and len(second) == 1

    def test_a_score_below_threshold_does_nothing(self) -> None:
        w, _, _, _ = build([0.49] * 40, threshold=0.5)
        assert feed_hops(w, 30) == []

    def test_disabled_means_no_work_at_all(self) -> None:
        # Not just "no events" -- no inference either. An always-on
        # detector that respects a config flag only at the last step is
        # still an always-on detector.
        w, mel, _, _ = build([0.99] * 40, enabled=False)
        primed = len(mel.calls)
        assert feed_hops(w, 10) == []
        assert len(mel.calls) == primed


class TestContextAndDiagnostics:
    def test_context_is_1_28_seconds(self) -> None:
        w, _, _, _ = build([0.0])
        assert w.context_s == pytest.approx(16 * 1280 / 16000.0)

    def test_diagnostics_answer_is_it_hearing_me_at_all(self) -> None:
        # A score that moves with your voice but tops out at 0.3 is a
        # threshold problem; one pinned at 0.0 while you shout is a
        # pipeline problem. They look identical without this.
        w, _, _, _ = build([0.31] * 40, threshold=0.9)
        feed_hops(w, 20)
        d = w.diagnostics()
        assert d["last_score"] == pytest.approx(0.31)
        assert d["threshold"] == pytest.approx(0.9)
        assert d["hops_scored"] > 0

    def test_reset_reprimes_rather_than_zeroing(self) -> None:
        # The context before real audio is the MEL OF SILENCE, not zeros.
        # After mel/10 + 2 those are different values, and a classifier
        # primed with impossible ones spends its first second in a state
        # it never saw in training.
        w, _, emb, _ = build([0.0] * 50)
        feed_hops(w, 20)
        w.reset()
        feed_hops(w, 2, start=100.0)
        window = emb.calls[-1]["input_1"]
        assert not np.any(window == 0.0), "reset left raw zeros in the mel context"


class TestConfig:
    def test_the_shapes_are_the_measured_ones(self) -> None:
        cfg = Elizabeth().wake
        assert (cfg.hop_samples, cfg.mel_window, cfg.embedding_window) == (1280, 76, 16)
        assert (cfg.mel_scale, cfg.mel_offset) == (10.0, 2.0)

    def test_the_head_path_is_ours_and_the_features_are_not(self) -> None:
        cfg = Elizabeth().wake
        assert cfg.features_dir == "models/openwakeword-features"
        assert "wake-hey-elizabeth" in cfg.head_path
