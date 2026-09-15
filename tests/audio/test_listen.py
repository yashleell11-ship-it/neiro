"""The hands-free listener, driven frame by frame with scripted
probabilities (src/elizabeth/audio/listen.py).

Every expectation is derived from config at test time — how many frames
the gate needs, how long the dead zone is — so a retuned config moves
the tests with it, and a broken machine cannot hide behind a number
that happens to match. Audio is checked by sample value against the
frames that were fed, never by length: a pre-roll of the right size
spliced from the wrong place is the failure mode that matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from elizabeth.audio.listen import (
    BARGEIN_RESUME_S,
    EventKind,
    Listener,
    ListenEvent,
    ListenState,
)
from elizabeth.audio.vad import VadFrameSizeError
from elizabeth.config import Elizabeth

T0 = 100.0  # the fake clock's reading at the first frame
Policy = Callable[[float], tuple[bool, float]]


def always(done: bool, p: float) -> Policy:
    return lambda waited_s: (done, p)


class Frames:
    """Frame counts the listener must agree with, all from config."""

    def __init__(self, cfg: Elizabeth) -> None:
        self.n = cfg.vad.frame_samples
        self.seconds = self.n / cfg.audio.input_samplerate
        ms = self.seconds * 1000.0
        self.start = math.ceil(cfg.vad.min_speech_ms / ms)
        self.stop = math.ceil(cfg.vad.min_silence_ms / ms)
        # BargeInDetector counts the frame it is fed before checking the
        # dead zone, so the first frame that can count is this one
        # (1-based), and the hold trips `hold` frames later.
        self.first_armed = math.ceil(cfg.vad.bargein_dead_zone_ms / ms)
        self.hold = math.ceil(cfg.vad.bargein_speech_ms / ms)
        self.max_wait = math.ceil(cfg.endpoint.max_wait_s / self.seconds)
        self.resume = math.ceil(BARGEIN_RESUME_S / self.seconds)
        self.pre_roll_samples = int(cfg.audio.pre_roll_s * cfg.audio.input_samplerate)

    def quiet(self, i: int) -> np.ndarray:
        """Below the echo floor, and unique per frame so a splice from
        the wrong place is caught by value."""
        return np.full(self.n, 1e-3 + i * 1e-6, dtype=np.float32)

    def loud(self, i: int) -> np.ndarray:
        return np.full(self.n, 0.3 + i * 1e-6, dtype=np.float32)


class Mic:
    """Feeds frames with a scripted Silero probability and a scripted
    smart-turn, on a clock that advances one frame per feed."""

    def __init__(self, cfg: Elizabeth, policy: Policy) -> None:
        self.cfg = cfg
        self.f = Frames(cfg)
        self.policy = policy
        self.t = T0
        self.p = 0.0
        self.asks: list[float] = []
        self.resets = 0
        self.fed: list[np.ndarray] = []
        self.events: list[tuple[int, ListenEvent]] = []
        self.listener = Listener(
            cfg,
            prob=lambda frame: self.p,
            complete=self._complete,
            reset=self._reset,
            clock=lambda: self.t,
        )

    def _complete(self, pcm: np.ndarray, waited_s: float) -> tuple[bool, float]:
        self.asks.append(waited_s)
        return self.policy(waited_s)

    def _reset(self) -> None:
        self.resets += 1

    @property
    def index(self) -> int:
        return len(self.fed)

    def time_of(self, index: int) -> float:
        return T0 + index * self.f.seconds

    def feed(self, frame: np.ndarray, p: float) -> ListenEvent | None:
        self.p = p
        event = self.listener.feed(frame)
        if event is not None:
            self.events.append((self.index, event))
        self.fed.append(frame)
        self.t += self.f.seconds
        return event

    def quiet(self, count: int, p: float = 0.0) -> None:
        for _ in range(count):
            self.feed(self.f.quiet(self.index), p)

    def loud(self, count: int, p: float = 0.99) -> None:
        for _ in range(count):
            self.feed(self.f.loud(self.index), p)

    def stream(self) -> np.ndarray:
        return np.concatenate(self.fed)

    def only(self, kind: EventKind) -> tuple[int, ListenEvent]:
        found = [(i, e) for i, e in self.events if e.kind is kind]
        assert len(found) == 1, f"expected one {kind}, got {self.events}"
        return found[0]

    def none_of(self, kind: EventKind) -> None:
        assert not [e for _, e in self.events if e.kind is kind], self.events

    def quiet_until(self, kind: EventKind, limit: int = 200) -> int:
        """Feed quiet frames one at a time until `kind` arrives; the
        next frame fed is the first one after it."""
        for _ in range(limit):
            event = self.feed(self.f.quiet(self.index), 0.0)
            if event is not None and event.kind is kind:
                return self.index - 1
        raise AssertionError(f"no {kind} within {limit} quiet frames: {self.events}")


@pytest.fixture
def cfg() -> Elizabeth:
    return Elizabeth()


LEAD = 10  # quiet frames before he speaks, more than one pre-roll's worth
SPEECH = 31  # ~1 s


class TestHisTurn:
    def test_the_happy_path_ends_at_the_first_pause_smart_turn_accepts(
        self, cfg: Elizabeth
    ) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.quiet(60)

        index, event = m.only(EventKind.UTTERANCE)
        # The endpoint is the frame the gate called a stop, and the
        # decision came from smart-turn, not the ceiling.
        assert index == LEAD + SPEECH + m.f.stop - 1
        assert event.t_endpoint == pytest.approx(m.time_of(index))
        assert event.t_speech_start == pytest.approx(m.time_of(LEAD))
        assert not event.forced
        assert event.probability == 0.9
        assert m.asks == [0.0]
        assert m.listener.state is ListenState.IDLE

    def test_the_audio_is_pre_roll_then_every_frame_up_to_the_decision(
        self, cfg: Elizabeth
    ) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.quiet(60)

        index, event = m.only(EventKind.UTTERANCE)
        first = LEAD * m.f.n - m.f.pre_roll_samples
        last = (index + 1) * m.f.n
        np.testing.assert_array_equal(event.audio, m.stream()[first:last])

    def test_the_pre_roll_is_the_audio_just_before_speech(self, cfg: Elizabeth) -> None:
        # By value: the samples in front of his first word are the ones
        # the mic heard right before it, not zeros and not older audio.
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.quiet(60)

        _, event = m.only(EventKind.UTTERANCE)
        before_speech = m.stream()[: LEAD * m.f.n]
        np.testing.assert_array_equal(
            event.audio[: m.f.pre_roll_samples], before_speech[-m.f.pre_roll_samples :]
        )
        # ...and the speech itself follows it without a gap.
        np.testing.assert_array_equal(
            event.audio[m.f.pre_roll_samples : m.f.pre_roll_samples + m.f.n],
            m.fed[LEAD],
        )

    def test_a_hesitation_smart_turn_rejects_does_not_end_the_utterance(
        self, cfg: Elizabeth
    ) -> None:
        # "I want to go to... uh... the library". A 600 ms pause is
        # well past the VAD stop; only smart-turn saying "not done"
        # keeps this one turn.
        pause = math.ceil(0.6 / Frames(cfg).seconds)
        answers: list[tuple[bool, float]] = []
        m = Mic(cfg, lambda waited_s: answers.pop(0))
        m.quiet(LEAD)
        m.loud(SPEECH)
        answers.extend([(False, 0.4)] * 10)
        m.quiet(pause)
        assert m.events == []
        # Smart-turn was consulted during the pause — at the stop, and
        # again per pause-length of further silence — and refused.
        assert len(m.asks) >= 2
        assert m.asks[0] == 0.0
        assert m.listener.state is ListenState.ENDPOINT_PENDING

        m.loud(15)
        assert m.listener.state is ListenState.SPEAKING
        answers[:] = [(True, 0.8)]
        m.quiet(60)

        index, event = m.only(EventKind.UTTERANCE)
        assert index == LEAD + SPEECH + pause + 15 + m.f.stop - 1
        assert event.t_speech_start == pytest.approx(m.time_of(LEAD))
        # One utterance, both halves in it.
        first = LEAD * m.f.n - m.f.pre_roll_samples
        np.testing.assert_array_equal(event.audio, m.stream()[first : (index + 1) * m.f.n])

    def test_max_delay_ends_a_turn_smart_turn_never_accepts(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(False, 0.3))
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.quiet(150)

        _, event = m.only(EventKind.UTTERANCE)
        assert event.forced
        stop = LEAD + SPEECH + m.f.stop - 1
        waited = event.t_endpoint - m.time_of(stop)
        assert cfg.endpoint.max_wait_s <= waited < cfg.endpoint.max_wait_s + 2 * m.f.seconds
        # The listener forced it; it did not lean on smart-turn's own
        # ceiling, and it kept asking until then.
        assert len(m.asks) > 1
        assert max(m.asks) < cfg.endpoint.max_wait_s
        assert event.probability == 0.3

    def test_one_loud_frame_starts_nothing(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(1)
        m.quiet(60)
        assert m.events == []
        assert m.asks == []

    def test_silero_is_reset_between_utterances(self, cfg: Elizabeth) -> None:
        # Or the tail of one utterance biases the start of the next.
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.quiet(60)
        assert m.resets == 1

    def test_a_wrong_frame_size_is_refused(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(True, 0.9))
        with pytest.raises(VadFrameSizeError):
            m.listener.feed(np.zeros(2 * m.f.n, dtype=np.float32))

    def test_recent_reads_what_was_just_fed(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        np.testing.assert_array_equal(
            m.listener.recent(2 * m.f.seconds), np.concatenate(m.fed[-2:])
        )


class TestHerTurn:
    def test_barge_in_waits_out_the_dead_zone_then_the_hold(self, cfg: Elizabeth) -> None:
        # Her own first syllable reaches the mic before any canceller
        # has adapted: loud, speech-like audio from the first frame of
        # her reply must not trip anything until the dead zone has
        # passed AND the hold has been sustained after it.
        m = Mic(cfg, always(True, 0.9))
        m.listener.set_speaking(True)
        m.loud(40)

        index, event = m.only(EventKind.BARGE_IN)
        assert index == m.f.first_armed + m.f.hold - 2
        # Speech-start is the first frame that counted, not the trip.
        assert event.t_speech_start == pytest.approx(m.time_of(m.f.first_armed - 1))
        assert m.listener.state is ListenState.SPEAKING
        assert not m.listener.her_turn

    def test_audio_under_the_echo_floor_never_interrupts_her(self, cfg: Elizabeth) -> None:
        # Silero can be fooled by her own residual echo; the RMS floor
        # cannot. Speech-probability 0.99 on near-silent frames is not
        # him.
        m = Mic(cfg, always(True, 0.9))
        m.listener.set_speaking(True)
        m.quiet(40, p=0.99)
        assert m.events == []

        m.loud(40)
        index, _ = m.only(EventKind.BARGE_IN)
        assert index == 40 + m.f.hold - 1

    def test_his_correction_starts_with_pre_roll_and_the_hold(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.listener.set_speaking(True)
        m.quiet(LEAD)
        m.loud(SPEECH)
        first_speech = LEAD
        m.quiet(60)

        _, barge = m.only(EventKind.BARGE_IN)
        index, event = m.only(EventKind.UTTERANCE)
        assert event.t_speech_start == barge.t_speech_start
        assert event.t_speech_start == pytest.approx(m.time_of(first_speech))
        first = first_speech * m.f.n - m.f.pre_roll_samples
        np.testing.assert_array_equal(event.audio, m.stream()[first : (index + 1) * m.f.n])

    def test_the_loop_confirming_she_stopped_does_not_drop_his_words(self, cfg: Elizabeth) -> None:
        # The natural call order: BARGE_IN, kill playback,
        # set_speaking(False). The utterance in progress survives it.
        m = Mic(cfg, always(True, 0.9))
        m.listener.set_speaking(True)
        m.quiet(LEAD)
        m.loud(SPEECH)
        m.only(EventKind.BARGE_IN)
        m.listener.set_speaking(False)
        assert m.listener.state is ListenState.SPEAKING
        m.quiet(60)
        m.only(EventKind.UTTERANCE)

    def test_set_speaking_is_idempotent(self, cfg: Elizabeth) -> None:
        # A loop that calls it per chunk must not re-arm the dead zone
        # every time, or barge-in never arms at all.
        m = Mic(cfg, always(True, 0.9))
        m.listener.set_speaking(True)
        m.loud(3)
        m.listener.set_speaking(True)
        m.loud(40)
        index, _ = m.only(EventKind.BARGE_IN)
        assert index == m.f.first_armed + m.f.hold - 2

    def test_her_starting_while_he_speaks_waits_for_his_endpoint(self, cfg: Elizabeth) -> None:
        m = Mic(cfg, always(True, 0.9))
        m.quiet(LEAD)
        m.loud(SPEECH)
        assert m.listener.state is ListenState.SPEAKING
        m.listener.set_speaking(True)
        m.quiet_until(EventKind.UTTERANCE)

        assert m.listener.state is ListenState.IDLE
        assert m.listener.her_turn
        # Her mode began at his endpoint, dead zone and all: loud
        # speech-like audio from the very next frame must wait it out.
        before = len(m.events)
        m.loud(m.f.first_armed + m.f.hold - 2)
        assert len(m.events) == before
        m.loud(1)
        assert m.events[-1][1].kind is EventKind.BARGE_IN


class TestFalseInterruption:
    def barge(self, cfg: Elizabeth, policy: Policy) -> Mic:
        m = Mic(cfg, policy)
        m.listener.set_speaking(True)
        m.quiet(m.f.first_armed + 1)
        m.loud(m.f.hold + 2)
        m.only(EventKind.BARGE_IN)
        return m

    def test_a_cough_resumes_her(self, cfg: Elizabeth) -> None:
        m = self.barge(cfg, always(False, 0.2))
        (barged, _) = m.events[0]
        m.quiet(120)

        index, _ = m.only(EventKind.RESUME)
        m.none_of(EventKind.UTTERANCE)
        assert index == barged + m.f.resume
        # Smart-turn was given its chance inside the window.
        assert len(m.asks) >= 1
        assert m.listener.state is ListenState.IDLE
        assert m.listener.her_turn
        assert m.resets == 1

    def test_after_resume_the_dead_zone_is_armed_again(self, cfg: Elizabeth) -> None:
        # She starts speaking again: her first syllable must be as safe
        # as it was the first time.
        m = self.barge(cfg, always(False, 0.2))
        resumed_at = m.quiet_until(EventKind.RESUME)
        m.loud(40)
        barges = [i for i, e in m.events if e.kind is EventKind.BARGE_IN]
        assert barges[-1] == resumed_at + 1 + m.f.first_armed + m.f.hold - 2

    def test_a_turn_accepted_inside_the_window_is_not_resumed(self, cfg: Elizabeth) -> None:
        m = self.barge(cfg, always(True, 0.9))
        m.quiet(120)
        m.only(EventKind.UTTERANCE)
        m.none_of(EventKind.RESUME)
        assert not m.listener.her_turn

    def test_sustained_speech_past_the_window_is_a_real_interruption(self, cfg: Elizabeth) -> None:
        # He is still talking when the window closes: whatever
        # smart-turn says later, she does not talk over him.
        m = self.barge(cfg, always(False, 0.2))
        m.loud(m.f.resume + 5)
        m.quiet(150)
        m.none_of(EventKind.RESUME)
        _, event = m.only(EventKind.UTTERANCE)
        assert event.forced

    def test_a_window_past_the_hard_ceiling_can_never_resume_her(
        self, cfg: Elizabeth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Why the default sits below max_wait_s: past it the forced
        # endpoint delivers the cough as an utterance first, and RESUME
        # is unreachable. Whoever moves the window into config needs
        # this to stay true.
        from elizabeth.audio import listen

        monkeypatch.setattr(listen, "BARGEIN_RESUME_S", cfg.endpoint.max_wait_s + 1.0)
        m = self.barge(cfg, always(False, 0.2))
        m.quiet(200)
        m.none_of(EventKind.RESUME)
        _, event = m.only(EventKind.UTTERANCE)
        assert event.forced


# -- against the real models -------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORPUS = REPO / "data/datasets/crema-d/extracted"


def real_speech() -> Path | None:
    """A CC0 fixture if one has been added, else a CREMA-D clip from the
    (gitignored) corpus, else nothing."""
    fixtures = sorted(FIXTURES.glob("*.wav")) if FIXTURES.is_dir() else []
    if fixtures:
        return fixtures[0]
    clips = sorted(CORPUS.rglob("1001_*.wav"))[:1] if CORPUS.is_dir() else []
    return clips[0] if clips else None


class TestAgainstTheModels:
    def test_a_real_clip_becomes_one_utterance_with_the_speech_in_it(self) -> None:
        cfg = Elizabeth()
        clip = real_speech()
        if clip is None:
            pytest.skip("no CC0 real-speech fixture under tests/audio/fixtures, and no CREMA-D")
        if not (REPO / cfg.vad.model_path).exists():
            pytest.skip("silero-vad not downloaded")
        if not (REPO / cfg.endpoint.model_path).exists():
            pytest.skip("smart-turn not downloaded")
        import soundfile as sf

        speech, sr = sf.read(clip, dtype="float32")
        assert sr == cfg.audio.input_samplerate
        f = Frames(cfg)
        lead = np.zeros(int(0.5 * sr), dtype=np.float32)
        # Enough trailing silence for the hard ceiling, should smart-turn
        # never accept it.
        tail = np.zeros(int((cfg.endpoint.max_wait_s + 1.0) * sr), dtype=np.float32)
        stream = np.concatenate([lead, speech, tail])
        stream = stream[: (stream.size // f.n) * f.n]

        listener = Listener.with_models(cfg, clock=lambda: 0.0)
        events: list[tuple[int, ListenEvent]] = []
        for i in range(stream.size // f.n):
            event = listener.feed(stream[i * f.n : (i + 1) * f.n])
            if event is not None:
                events.append((i, event))
        utterances = [(i, e) for i, e in events if e.kind is EventKind.UTTERANCE]
        assert utterances, f"no utterance from {clip.name}: {events}"
        _, first = utterances[0]
        # The loudest moment of the clip is inside what was captured.
        assert float(np.abs(first.audio).max()) == float(np.abs(speech).max())
        assert listener.state is ListenState.IDLE
