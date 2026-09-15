"""Hands-free listening — the Stage 2 state machine that replaces the spacebar.

`elizabeth talk` ends an utterance when SPACE is pressed. This is the thing
that presses it: Silero says he stopped making noise, smart-turn says
whether he finished a *turn*, and while she is speaking a stricter rule
decides whether he is talking over her. Nothing here touches a
microphone, a model or the event loop — it is fed one frame at a time
and hands back an event, so every path is driven by scripted
probabilities in tests and by the real models in the loop.

**Integration** (talk.py, or its Stage 2 successor, owns this):

  - Build one with `Listener.with_models(cfg)` — real Silero and
    smart-turn, or `Listener(cfg, prob=..., complete=...)` with fakes.
  - For every 512-sample float32 block the capture callback produces
    (`cfg.audio.input_blocksize`, exactly one Silero window), call
    `feed(frame)` from the loop thread, in order, never skipping one:
    Silero is a stateful RNN and the hysteresis counts frames. `feed`
    returns a `ListenEvent` or None. `complete()` blocks for one
    smart-turn call (~50 ms on this laptop's CPU) at most once per
    `cfg.vad.min_silence_ms` of silence, so feed from the asyncio side
    of a queue, not from inside the PortAudio callback.
  - `ListenEvent.kind`:
      UTTERANCE  — `audio` is the whole utterance (pre-roll included),
                   `t_speech_start` / `t_endpoint` are the clock at his
                   first speech frame and at the decision: hand them to
                   `Turn.t0_speech_start` / `Turn.t_endpoint` and the
                   audio to `Daemon.handle`. `forced` says the hard
                   ceiling ended it, not smart-turn; `probability` is
                   smart-turn's last answer.
      BARGE_IN   — he is talking over her: `Daemon.interrupt()`, kill
                   playback. The listener is already capturing his
                   correction (the frames that tripped the rule are the
                   start of it), so the UTTERANCE follows on its own.
      RESUME     — the BARGE_IN was a cough or a door: nothing usable
                   followed it inside `bargein_resume_s`. Resume her
                   reply where it stopped.
  - Tell it when she is audible: `set_speaking(True)` the moment
    playback starts, `set_speaking(False)` when it ends. Both are
    idempotent and safe to call after a BARGE_IN or RESUME — the
    listener already switched modes when it emitted them.
  - `recent(seconds)` reads the same ring the pre-roll comes from, so
    the affect window is pulled from here while `state != IDLE`.

**Why a machine and not two detectors.** `vad.py` already has the
speech gate and the barge-in detector; what it deliberately does not
have is the *meaning* of their transitions. A VAD stop is a dozen times
true inside one sentence, and acting on it is what cuts people off at
"I want to go to... uh... the library". So the stop only opens
ENDPOINT_PENDING; smart-turn is asked, once per pause-length of
further silence, and a "no" keeps the utterance open until he either
resumes (back to SPEAKING, same utterance) or the hard ceiling
`cfg.endpoint.max_wait_s` ends it. Every number is the one the two
detectors already use, so a change in config moves both.

**Why the pre-roll comes from a ring.** Speech-start is only known
`min_speech_ms` after it happened, and the first syllable was before
even that. The ring holds the last `ring_buffer_seconds`; the utterance
is cut from it at speech-start — `pre_roll_s` plus the frames that
proved it was speech — exactly the Task 4 trick, without the key.

**Why barge-in has two gates and a floor.** A false positive cuts her
off mid-sentence; a false negative costs him a repeat. So while she is
audible the ordinary gate is not consulted at all: `BargeInDetector`
(dead zone, then a longer hold at a higher probability) decides, and a
frame only counts toward the hold if its RMS is above the echo floor —
her own voice, back through the speakers profile, must not be what
interrupts her. A brief noise that trips it anyway is undone by RESUME
rather than by an apology: if smart-turn has not accepted what followed
as a turn inside the window, and he is not still talking, she carries
on. Sustained speech past the window confirms the interruption.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from elizabeth.audio.ring import RingBuffer
from elizabeth.audio.vad import BargeInDetector, SpeechGate, VadFrameSizeError
from elizabeth.config import Elizabeth

log = logging.getLogger(__name__)

# Two tunables VadConfig does not have yet; read via getattr so a config
# that grows them wins, and listed here so they can move into config.py.
#
# A frame only counts toward the barge-in hold if its RMS is above this.
# A per-machine number: residual echo through the speakers profile
# after PipeWire's canceller, which `elizabeth doctor` should measure
# (aec.py::suppression_db) — 0.01 is -40 dBFS, above digital silence and
# a quiet room, and below any voice aimed at the mic. Earbuds never echo,
# so it costs that profile nothing.
BARGEIN_ECHO_FLOOR_RMS = 0.01
# How long after a BARGE_IN she waits for something usable before
# resuming. Must exceed `min_silence_ms` plus one smart-turn call (a real
# one-word interruption is accepted in ~0.3 s) and stay below
# `endpoint.max_wait_s`, past which the forced endpoint would deliver the
# cough as an utterance and RESUME could never fire.
BARGEIN_RESUME_S = 1.5


class ListenState(StrEnum):
    IDLE = "idle"  # nothing of his in progress (she may be speaking)
    SPEAKING = "speaking"  # his utterance is being captured
    ENDPOINT_PENDING = "endpoint_pending"  # VAD stopped; smart-turn deciding


class EventKind(StrEnum):
    UTTERANCE = "utterance"
    BARGE_IN = "barge_in"
    RESUME = "resume"


@dataclass(frozen=True)
class ListenEvent:
    kind: EventKind
    t_speech_start: float | None = None
    t_endpoint: float | None = None
    audio: np.ndarray | None = None
    forced: bool = False  # UTTERANCE: max_wait_s ended it, not smart-turn
    probability: float | None = None  # UTTERANCE: smart-turn's last answer


class Listener:
    """One frame in, at most one event out. protocols-shaped by hand:
    `prob(frame) -> float` is Silero, `complete(pcm, waited_s) ->
    (bool, float)` is smart-turn, `reset()` clears Silero's RNN between
    utterances, and `clock()` stamps the events.
    """

    def __init__(
        self,
        cfg: Elizabeth | None = None,
        *,
        prob: Callable[[np.ndarray], float],
        complete: Callable[[np.ndarray, float], tuple[bool, float]],
        reset: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg or Elizabeth()
        self._prob = prob
        self._complete = complete
        self._reset_prob = reset
        self._clock = clock

        vad = self.cfg.vad
        self._frame_samples = vad.frame_samples
        self._frame_s = vad.frame_samples / self.cfg.audio.input_samplerate
        frame_ms = self._frame_s * 1000.0
        self._gate = SpeechGate(cfg=self.cfg, frame_ms=frame_ms)
        self._bargein = BargeInDetector(cfg=self.cfg, frame_ms=frame_ms)
        # The same frame counts the two detectors use, so the listener's
        # own bookkeeping cannot disagree with them.
        self._stop_frames = math.ceil(vad.min_silence_ms / frame_ms)
        self._max_wait_frames = math.ceil(self.cfg.endpoint.max_wait_s / self._frame_s)
        self._echo_floor = float(getattr(vad, "bargein_echo_floor_rms", BARGEIN_ECHO_FLOOR_RMS))
        self._resume_frames = math.ceil(
            float(getattr(vad, "bargein_resume_s", BARGEIN_RESUME_S)) / self._frame_s
        )

        # The ring must hold the pre-roll plus the longest run of frames
        # that proves speech before speech-start is known, and the affect
        # window if the loop reads it from here.
        longest_run_s = max(vad.min_speech_ms, vad.bargein_speech_ms) / 1000.0 + self._frame_s
        self._ring = RingBuffer(
            capacity_seconds=max(
                self.cfg.audio.ring_buffer_seconds,
                self.cfg.audio.pre_roll_s + longest_run_s,
                self.cfg.affect.window_seconds,
            ),
            samplerate=self.cfg.audio.input_samplerate,
        )

        self.state = ListenState.IDLE
        self.her_turn = False
        self._frames: list[np.ndarray] = []  # his utterance, pre-roll first
        self._t_speech_start: float | None = None
        # The current run of frames that count as speech (either gate),
        # so speech-start can be dated to its first frame.
        self._run_frames = 0
        self._run_t0: float | None = None
        self._pending_frames = 0
        self._since_ask = 0
        self._last_p: float | None = None
        # Frames since the BARGE_IN whose RESUME is still possible; None
        # once it is confirmed, answered, or undone.
        self._since_bargein: int | None = None

    @classmethod
    def with_models(cls, cfg: Elizabeth | None = None, *, warm: bool = True, **kwargs) -> Listener:
        """The real thing: Silero behind `prob`, smart-turn behind
        `complete`, Silero's `reset` between utterances.

        Warmed by default: measured on this laptop, the first smart-turn
        call through a cold librosa costs ~2.9 s against ~50 ms after —
        and it would land on the first pause of his first sentence.
        """
        from elizabeth.audio.endpoint import SmartTurnEndpointer
        from elizabeth.audio.vad import SileroVad

        cfg = cfg or Elizabeth()
        vad = SileroVad(cfg)
        endpointer = SmartTurnEndpointer(cfg=cfg)
        endpointer.load()
        if warm:
            vad.warm()
            endpointer.warm()
        return cls(
            cfg, prob=vad.probability, complete=endpointer.is_complete, reset=vad.reset, **kwargs
        )

    # -- what the loop tells it ------------------------------------------

    def set_speaking(self, speaking: bool) -> None:
        """She has started (True) or finished (False) being audible.

        Idempotent on purpose: a loop that calls this per chunk must not
        re-arm the dead zone each time, or barge-in never arms. His
        utterance in progress is never dropped by either call — if she
        starts while he is mid-sentence, his endpoint still fires, and
        her mode begins after it.
        """
        if speaking == self.her_turn:
            return
        self.her_turn = speaking
        if self.state is not ListenState.IDLE:
            return
        if speaking:
            self._arm_bargein()
        else:
            self._bargein.stop_speaking()
            self._gate.reset()
            self._clear_run()

    def recent(self, seconds: float) -> np.ndarray:
        """The last `seconds` of everything fed, for the affect window."""
        return self._ring.read_last(seconds)

    # -- one frame ---------------------------------------------------------

    def feed(self, frame: np.ndarray) -> ListenEvent | None:
        audio = np.asarray(frame, dtype=np.float32).reshape(-1)
        if audio.size != self._frame_samples:
            raise VadFrameSizeError(
                f"Listener needs exactly {self._frame_samples} samples per frame "
                f"(one Silero window), got {audio.size}."
            )
        now = self._clock()
        self._ring.write(audio)
        if self.state is not ListenState.IDLE:
            self._frames.append(audio)
        p = float(self._prob(audio))

        if self.state is ListenState.IDLE and self.her_turn:
            return self._her_turn_frame(audio, p, now)
        return self._his_turn_frame(p, now)

    # -- her turn: the barge-in rule --------------------------------------

    def _her_turn_frame(self, audio: np.ndarray, p: float, now: float) -> ListenEvent | None:
        # A frame quieter than the echo floor is not evidence of him,
        # whatever Silero makes of it: fed to the detector as silence so
        # it breaks the hold exactly as a quiet frame would.
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        heard = p if rms >= self._echo_floor else 0.0
        tripped = self._bargein.update(heard)
        self._track_run(self._bargein.armed and heard >= self.cfg.vad.bargein_probability, now)
        if not tripped:
            return None

        # Everything the hold counted is the start of his correction.
        t_start = self._run_t0
        self._begin_utterance(t_start, self._run_frames)
        self.her_turn = False
        self._since_bargein = 0
        log.debug("barge-in at %.3f", now)
        return ListenEvent(kind=EventKind.BARGE_IN, t_speech_start=t_start)

    # -- his turn: speech, then the endpoint ------------------------------

    def _his_turn_frame(self, p: float, now: float) -> ListenEvent | None:
        transition = self._gate.update(p)
        self._track_run(p >= self.cfg.vad.threshold, now)

        if self._since_bargein is not None:
            self._since_bargein += 1
            if self._since_bargein >= self._resume_frames:
                if self.state is ListenState.SPEAKING:
                    # Still talking when the window closed: a real
                    # interruption, whatever smart-turn says later.
                    self._since_bargein = None
                else:
                    return self._resume(now)

        if self.state is ListenState.IDLE:
            if transition == "started":
                self._begin_utterance(self._run_t0, self._run_frames)
            return None

        if self.state is ListenState.SPEAKING:
            if transition == "stopped":
                self.state = ListenState.ENDPOINT_PENDING
                self._pending_frames = 0
                self._since_ask = 0
                return self._ask(now)
            return None

        # ENDPOINT_PENDING
        if transition == "started":
            # He resumed inside the pause: the same utterance goes on.
            self.state = ListenState.SPEAKING
            return None
        self._pending_frames += 1
        self._since_ask += 1
        if self._run_frames:
            # He sounds like he is resuming and the gate has not confirmed
            # it yet. Neither asking nor forcing on a frame of speech can
            # be right; a quiet frame or the gate settles it within
            # `min_speech_ms`.
            return None
        if self._pending_frames >= self._max_wait_frames:
            return self._utterance(now, forced=True)
        if self._since_ask >= self._stop_frames:
            self._since_ask = 0
            return self._ask(now)
        return None

    def _ask(self, now: float) -> ListenEvent | None:
        waited_s = self._pending_frames * self._frame_s
        done, self._last_p = self._complete(np.concatenate(self._frames), waited_s)
        if done:
            return self._utterance(now, forced=False)
        return None

    # -- transitions -------------------------------------------------------

    def _begin_utterance(self, t_start: float | None, run_frames: int) -> None:
        self.state = ListenState.SPEAKING
        self._t_speech_start = t_start
        # The gate is what ends the utterance, so it must know one is
        # open — after a barge-in it was never consulted, and a cough
        # that stopped before it caught up would leave SPEAKING forever.
        self._gate.reset()
        self._gate.speaking = True
        # The frames that proved it was speech are already in the ring;
        # take them back out with the pre-roll in front.
        seconds = self.cfg.audio.pre_roll_s + run_frames * self._frame_s
        self._frames = [self._ring.read_last(seconds)]
        self._last_p = None

    def _utterance(self, now: float, *, forced: bool) -> ListenEvent:
        event = ListenEvent(
            kind=EventKind.UTTERANCE,
            t_speech_start=self._t_speech_start,
            t_endpoint=now,
            audio=np.concatenate(self._frames),
            forced=forced,
            probability=self._last_p,
        )
        log.debug(
            "utterance %.2fs, %s (p=%s)",
            event.audio.size / self.cfg.audio.input_samplerate,
            "forced" if forced else "smart-turn",
            self._last_p,
        )
        self._since_bargein = None
        self._to_idle()
        return event

    def _resume(self, now: float) -> ListenEvent:
        log.debug("false interruption: resuming her at %.3f", now)
        self._since_bargein = None
        self.her_turn = True
        self._to_idle()
        return ListenEvent(kind=EventKind.RESUME)

    def _to_idle(self) -> None:
        self.state = ListenState.IDLE
        self._frames = []
        self._t_speech_start = None
        self._pending_frames = 0
        self._since_ask = 0
        self._gate.reset()
        self._clear_run()
        if self._reset_prob is not None:
            # Or the tail of this utterance biases the start of the next.
            self._reset_prob()
        if self.her_turn:
            self._arm_bargein()

    def _arm_bargein(self) -> None:
        self._bargein.start_speaking()
        self._gate.reset()
        self._clear_run()

    def _track_run(self, counts: bool, now: float) -> None:
        if not counts:
            self._clear_run()
            return
        if self._run_frames == 0:
            self._run_t0 = now
        self._run_frames += 1

    def _clear_run(self) -> None:
        self._run_frames = 0
        self._run_t0 = None
