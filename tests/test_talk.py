"""`elizabeth talk` driven end to end with fakes: keys in, a WAV and a
playback call out, and the spacebar as barge-in.

The daemon and orchestrator are real; only the providers, the
microphone, the keyboard and the speaker are faked. That is the point:
until this command existed the loop was only ever assembled by tests,
and a loop that is only assembled by tests is not a product.
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from elizabeth.config import Elizabeth
from elizabeth.daemon import Daemon
from elizabeth.state import Locality
from elizabeth.talk import TalkLoop

SR = 16000


class FakeCapture:
    def __init__(self, utterance: np.ndarray) -> None:
        self.utterance = utterance
        self.armed = False
        self.level = 0.0
        self.arms = 0

    def arm(self) -> None:
        self.armed = True
        self.arms += 1

    def disarm(self) -> np.ndarray:
        self.armed = False
        return self.utterance

    def recent(self, seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * SR), dtype=np.float32)


class FakeStt:
    locality = Locality.LOCAL_PINNED

    def __init__(self) -> None:
        self.heard: list[int] = []

    async def transcribe(self, pcm_16k: np.ndarray) -> str:
        self.heard.append(int(pcm_16k.size))
        return "what time is it"


class ScriptedLlm:
    """First call streams slowly (so a barge-in can land mid-reply);
    every later call answers at once."""

    locality = Locality.TIERABLE

    def __init__(self, slow_first: bool = False) -> None:
        self.calls = 0
        self.slow_first = slow_first

    async def stream(self, messages, tools=None):
        self.calls += 1
        if self.slow_first and self.calls == 1:
            yield {"text": "<e:happy:5>"}
            for _ in range(400):
                await asyncio.sleep(0.01)
                yield {"text": "word "}
            return
        yield {"text": "<e:happy:5>It is noon. Go eat something."}


class FakeTts:
    locality = Locality.LOCAL_PINNED

    async def synth(self, text, state):
        yield np.full(2400, 0.1, dtype=np.float32), None


class FakePlayer:
    def __init__(self) -> None:
        self.started: list[Path] = []
        self.terminated = 0
        self._done = False

    def __call__(self, path: Path) -> FakePlayer:
        self.started.append(path)
        return self

    def terminate(self) -> None:
        self.terminated += 1
        self._done = True

    def poll(self) -> int | None:
        return 0 if self._done else None


class MonitoringCapture(FakeCapture):
    """A capture with the wake word's tap on it.

    `monitor()` returns fresh audio exactly once, the way the real one
    does — returning the same block forever would let a test pass while
    the loop re-scored the same second of audio on every tick.
    """

    def __init__(self, utterance: np.ndarray, blocks: list[np.ndarray] | None = None,
                 dropped: bool = False) -> None:
        super().__init__(utterance)
        self.blocks = deque(blocks or [])
        self.dropped = dropped
        self.monitor_calls = 0

    def monitor(self) -> tuple[np.ndarray, bool]:
        self.monitor_calls += 1
        if self.blocks:
            return self.blocks.popleft(), self.dropped
        return np.zeros(0, dtype=np.float32), self.dropped


class ScriptedWake:
    """Fires on the Nth non-empty block it is fed."""

    def __init__(self, fire_on: int = 1) -> None:
        self.fire_on = fire_on
        self.fed = 0
        self.resets = 0
        self.samples_seen = 0

    def feed(self, audio, *, now: float):
        from elizabeth.audio.wake import WakeEvent

        self.fed += 1
        self.samples_seen += int(audio.size)
        if self.fed == self.fire_on:
            return WakeEvent(score=0.93, t_detected=now, context_s=1.28)
        return None

    def reset(self) -> None:
        self.resets += 1


def _loop(tmp_path: Path, keys: list[str | None], llm: ScriptedLlm,
          capture=None, wake=None) -> tuple[TalkLoop, list[str]]:
    cfg = Elizabeth()
    daemon = Daemon(cfg)
    daemon.build(stt=FakeStt(), llm=llm, tts=FakeTts())
    script = deque(keys)
    lines: list[str] = []
    loop = TalkLoop(
        daemon=daemon,
        cfg=cfg,
        capture=capture or FakeCapture(np.full(SR, 0.05, dtype=np.float32)),
        read_key=lambda: script.popleft() if script else None,
        play=FakePlayer(),
        out=lines.append,
        reply_path=tmp_path / "reply.wav",
        wake=wake,
    )
    return loop, lines


class TestOneTurn:
    def test_space_space_produces_a_reply_and_plays_it(self, tmp_path: Path) -> None:
        loop, lines = _loop(tmp_path, [" ", " "], ScriptedLlm())
        assert asyncio.run(loop.run(max_turns=1)) == 0

        assert loop.turns_done == 1
        result = loop.results[0]
        assert result.error is None and not result.cancelled
        assert result.transcript == "what time is it"
        assert "It is noon" in result.reply
        # The whole utterance reached STT, not just what came after arming.
        assert loop.daemon.orchestrator.stt.heard == [SR]
        # A WAV was written and handed to the player.
        assert loop.play.started == [tmp_path / "reply.wav"]
        assert (tmp_path / "reply.wav").stat().st_size > 44
        assert any(line.startswith("elizabeth  ") for line in lines)
        assert any(line.startswith("turn ") for line in lines), "no HUD line"

    def test_the_tag_never_reaches_the_printed_reply(self, tmp_path: Path) -> None:
        loop, lines = _loop(tmp_path, [" ", " "], ScriptedLlm())
        asyncio.run(loop.run(max_turns=1))
        spoken = next(line for line in lines if line.startswith("elizabeth  "))
        assert "<e:" not in spoken
        assert loop.results[0].state.label.value == "happy"

    def test_q_quits_without_a_turn(self, tmp_path: Path) -> None:
        loop, _ = _loop(tmp_path, ["q"], ScriptedLlm())
        assert asyncio.run(loop.run()) == 0
        assert loop.turns_done == 0


class TestBargeIn:
    def test_space_while_she_speaks_cancels_and_starts_the_next_turn(self, tmp_path: Path) -> None:
        # arm, end (slow reply starts) … three idle polls … arm (barge-in), end.
        keys: list[str | None] = [" ", " ", None, None, None, " ", " "]
        loop, lines = _loop(tmp_path, keys, ScriptedLlm(slow_first=True))
        asyncio.run(loop.run(max_turns=2))

        assert loop.interrupted == 1
        first, second = loop.results
        assert first.cancelled, "the slow reply was not cancelled"
        assert not second.cancelled and second.error is None
        assert "It is noon" in second.reply
        assert "(interrupted)" in lines
        # The cancelled reply was never played.
        assert loop.play.started == [tmp_path / "reply.wav"]
        assert loop.daemon.orchestrator.llm.calls == 2

    def test_space_during_playback_kills_the_player(self, tmp_path: Path) -> None:
        keys: list[str | None] = [" ", " ", None, None, None, None, " ", "q"]
        loop, _ = _loop(tmp_path, keys, ScriptedLlm())
        asyncio.run(loop.run())
        assert loop.play.started, "nothing was played"
        assert loop.play.terminated >= 1


class TestAffectIsObservedWhileSpeaking:
    def test_observe_runs_before_the_endpoint(self, tmp_path: Path, monkeypatch) -> None:
        # The annotation is ready at the endpoint and costs the turn
        # nothing — but only if the loop actually feeds the window in.
        keys: list[str | None] = [" ", None, " "]
        loop, _ = _loop(tmp_path, keys, ScriptedLlm())
        seen: list[int] = []

        async def observe(window: np.ndarray) -> str | None:
            seen.append(int(window.size))
            return None

        monkeypatch.setattr(loop.daemon, "observe", observe)
        asyncio.run(loop.run(max_turns=1))
        assert seen and seen[0] == int(loop.cfg.affect.window_seconds * SR)


class TestPhase:
    def test_phase_follows_the_loop(self, tmp_path: Path) -> None:
        # The display shows exactly one of four words; each must come
        # from the loop's real state, not from a flag set beside it.
        loop, _ = _loop(tmp_path, [], ScriptedLlm(slow_first=True))
        assert loop.phase == "listening"

        async def go() -> list[str]:
            phases = []
            await loop.toggle()  # arm
            phases.append(loop.phase)
            await loop.toggle()  # end -> slow turn in flight
            await asyncio.sleep(0.05)
            phases.append(loop.phase)
            await loop.toggle()  # barge-in: cancels, arms again
            phases.append(loop.phase)
            await loop.toggle()  # end -> quick turn, then playback
            await asyncio.wait_for(loop._inflight, 2.0)
            loop._reap()
            phases.append(loop.phase)
            await loop._stop_everything()
            return phases

        assert asyncio.run(go()) == ["recording", "thinking", "recording", "speaking"]


class FakeSink:
    """A sink with no file and no browser: it records what it was given."""

    def __init__(self) -> None:
        self.plays: list[tuple[int, str]] = []
        self.cancels = 0
        self.closed = 0

    async def play(self, turn, pcm, seq=0, text="", visemes=None) -> None:
        self.plays.append((seq, text))

    async def cancel(self, turn) -> None:
        self.cancels += 1

    def close(self) -> None:
        self.closed += 1


class FakeBrowserSink(FakeSink):
    """Shaped like the WebSocket sink after a turn: it is the playback."""

    def __init__(self) -> None:
        super().__init__()
        self.terminated = 0
        self.played_waits: list[float] = []

    def terminate(self) -> None:
        self.terminated += 1

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    async def wait_played(self, timeout: float) -> bool:
        self.played_waits.append(timeout)
        return True


class TestBrowserWiring:
    """`--browser` is a `sink_factory` and a `server`; the loop itself
    does not change. These drive it with fakes of both."""

    def test_the_factory_decides_what_a_turn_plays_through(self, tmp_path: Path) -> None:
        made: list[FakeSink] = []

        def factory() -> FakeSink:
            made.append(FakeSink())
            return made[-1]

        loop, lines = _loop(tmp_path, [" ", " "], ScriptedLlm())
        loop.sink_factory = factory
        assert asyncio.run(loop.run(max_turns=1)) == 0

        assert len(made) == 1
        assert loop.daemon.orchestrator.sink is made[0]
        # Both sentences of the reply went to the fake, in order.
        assert [seq for seq, _ in made[0].plays] == [0, 1]
        assert made[0].closed == 1
        # Nothing was written or handed to paplay: the sink played it.
        assert loop.play.started == []
        assert not (tmp_path / "reply.wav").exists()
        assert any(line.startswith("turn ") for line in lines), "no HUD line"

    def test_a_fresh_sink_per_turn(self, tmp_path: Path) -> None:
        made: list[FakeSink] = []

        def factory() -> FakeSink:
            made.append(FakeSink())
            return made[-1]

        keys: list[str | None] = [" ", " ", None, None, " ", " "]
        loop, _ = _loop(tmp_path, keys, ScriptedLlm())
        loop.sink_factory = factory
        asyncio.run(loop.run(max_turns=2))
        assert len(made) == 2 and made[0] is not made[1]

    def test_the_finished_browser_sink_stands_in_for_the_player(self, tmp_path: Path) -> None:
        # SPACE during the tail of a reply kills paplay for the file
        # sink; for the browser sink it must reach the same call.
        sink = FakeBrowserSink()
        loop, _ = _loop(tmp_path, [], ScriptedLlm())
        loop.sink_factory = lambda: sink

        async def go() -> list[str]:
            phases = []
            await loop.toggle()
            await loop.toggle()
            await asyncio.wait_for(loop._inflight, 2.0)
            loop._reap()
            phases.append(loop.phase)
            await loop.toggle()  # barge-in on the tail
            phases.append(loop.phase)
            await loop._stop_everything()
            return phases

        assert asyncio.run(go()) == ["speaking", "recording"]
        assert sink.terminated == 1
        # The HUD waited (bounded) for the browser's `played` before
        # printing, so the line can carry the real number.
        assert sink.played_waits and all(t > 0 for t in sink.played_waits)

    def test_the_server_runs_beside_the_loop_and_stops_with_it(self, tmp_path: Path) -> None:
        events: list[str] = []
        loop, _ = _loop(tmp_path, [" ", " "], ScriptedLlm())
        # The first key is read only after the server task exists.
        real_read = loop.read_key

        def read_key() -> str | None:
            events.append("key")
            return real_read()

        async def server() -> None:
            events.append("up")
            try:
                await asyncio.Event().wait()
            finally:
                events.append("down")

        loop.read_key = read_key
        loop.server = server
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert events[0] == "up" and events[-1] == "down"
        assert "key" in events

    def test_a_dead_server_ends_the_loop_with_an_error(self, tmp_path: Path) -> None:
        async def server() -> None:
            raise RuntimeError("port taken")

        loop, lines = _loop(tmp_path, [None, None, None], ScriptedLlm())
        loop.server = server
        assert asyncio.run(loop.run()) == 1
        assert any("face server stopped" in line and "port taken" in line for line in lines)

    def test_browser_options_bind_a_face_and_make_websocket_sinks(self) -> None:
        from elizabeth.audio.sink_ws import WsSink
        from elizabeth.talk import browser_options

        lines: list[str] = []
        cfg = Elizabeth()
        face, options = browser_options(cfg, out=lines.append, port=0)
        try:
            assert lines == [f"face: {face.url}"]
            assert str(face.port) in face.url and face.port != 0
            sink = options["sink_factory"]()
            assert isinstance(sink, WsSink)
            assert sink.session is face.session
            assert options["server"] == face.serve
            # Bound now, not later: the port is already taken.
            from elizabeth.server import Face

            with pytest.raises(OSError):
                Face(cfg, port=face.port).bind()
        finally:
            face.close()


class TestWakeWord:
    """Hands-free listening, and the four ways it must not break talking.

    The wake word is allowed to fail — an untrained head, a missing file,
    a stalled loop. None of those may cost the spacebar, and none of them
    may let the detector score a stream with a hole in it, which is how a
    detector starts firing on sentences nobody said.
    """

    BLOCK = np.full(1280, 0.05, dtype=np.float32)

    def test_her_name_starts_a_turn_without_a_key(self, tmp_path: Path) -> None:
        capture = MonitoringCapture(np.full(SR, 0.05, dtype=np.float32),
                                    blocks=[self.BLOCK, self.BLOCK])
        wake = ScriptedWake(fire_on=1)
        loop, lines = _loop(tmp_path, [None, None, " "], ScriptedLlm(),
                            capture=capture, wake=wake)
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert loop.woke == 1
        assert loop.turns_done == 1, "the wake word must start a real turn, not just arm"
        assert any("heard you" in line for line in lines)

    def test_the_spacebar_still_works_with_no_wake_word(self, tmp_path: Path) -> None:
        # The whole point of the degradation path: an untrained head
        # leaves `wake=None`, and that must be the loop as it always was.
        loop, _ = _loop(tmp_path, [" ", " "], ScriptedLlm(), wake=None)
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert loop.turns_done == 1 and loop.woke == 0

    def test_a_dropped_block_resets_the_context_instead_of_scoring_it(
        self, tmp_path: Path
    ) -> None:
        # A gap in the stream makes the previous two seconds a lie. Scoring
        # across it is how a detector fires on a sentence nobody spoke.
        capture = MonitoringCapture(np.full(SR, 0.05, dtype=np.float32),
                                    blocks=[self.BLOCK], dropped=True)
        wake = ScriptedWake(fire_on=1)
        loop, _ = _loop(tmp_path, [None, None, " ", " "], ScriptedLlm(),
                        capture=capture, wake=wake)
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert wake.fed == 0, "scored audio that had a gap in it"
        assert wake.resets >= 1
        assert loop.woke == 0

    def test_nothing_is_scored_while_the_mic_is_armed(self, tmp_path: Path) -> None:
        # She is already recording; the audio is the question, not the name.
        capture = MonitoringCapture(np.full(SR, 0.05, dtype=np.float32),
                                    blocks=[self.BLOCK] * 6)
        wake = ScriptedWake(fire_on=99)
        loop, _ = _loop(tmp_path, [" ", None, None, " "], ScriptedLlm(),
                        capture=capture, wake=wake)
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert wake.resets >= 1, "armed ticks must clear the context, not score it"

    def test_the_queue_is_drained_even_with_no_detector(self, tmp_path: Path) -> None:
        # Otherwise a run with the wake word off slowly fills a list that
        # nobody ever reads.
        capture = MonitoringCapture(np.full(SR, 0.05, dtype=np.float32),
                                    blocks=[self.BLOCK] * 3)
        loop, _ = _loop(tmp_path, [" ", " "], ScriptedLlm(), capture=capture, wake=None)
        assert asyncio.run(loop.run(max_turns=1)) == 0
        assert capture.monitor_calls > 0
