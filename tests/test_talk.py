"""`neiro talk` driven end to end with fakes: keys in, a WAV and a
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

from neiro.config import Neiro
from neiro.daemon import Daemon
from neiro.state import Locality
from neiro.talk import TalkLoop

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


def _loop(tmp_path: Path, keys: list[str | None], llm: ScriptedLlm) -> tuple[TalkLoop, list[str]]:
    cfg = Neiro()
    daemon = Daemon(cfg)
    daemon.build(stt=FakeStt(), llm=llm, tts=FakeTts())
    script = deque(keys)
    lines: list[str] = []
    loop = TalkLoop(
        daemon=daemon,
        cfg=cfg,
        capture=FakeCapture(np.full(SR, 0.05, dtype=np.float32)),
        read_key=lambda: script.popleft() if script else None,
        play=FakePlayer(),
        out=lines.append,
        reply_path=tmp_path / "reply.wav",
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
        assert any(line.startswith("neiro  ") for line in lines)
        assert any(line.startswith("turn ") for line in lines), "no HUD line"

    def test_the_tag_never_reaches_the_printed_reply(self, tmp_path: Path) -> None:
        loop, lines = _loop(tmp_path, [" ", " "], ScriptedLlm())
        asyncio.run(loop.run(max_turns=1))
        spoken = next(line for line in lines if line.startswith("neiro  "))
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
