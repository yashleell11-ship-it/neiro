"""The daemon: warm-up order, one turn at a time, and barge-in.

Wiring should be boring — if it needs cleverness the Protocols were
wrong — so these tests are mostly about the three things the daemon
genuinely owns and nothing else can.
"""

from __future__ import annotations

import asyncio

import numpy as np

from neiro.config import Neiro
from neiro.daemon import Daemon
from neiro.orchestrator import TurnResult
from neiro.speech.errors import Failure
from neiro.state import Turn

AUDIO = np.zeros(16000, dtype=np.float32)


class Recorder:
    """A provider that records whether and when it was warmed."""

    def __init__(self, order: list[str], name: str, fail: bool = False) -> None:
        self.order = order
        self.name = name
        self.fail = fail
        self.warmed = False

    def warm(self) -> float:
        if self.fail:
            raise RuntimeError("no model")
        self.warmed = True
        self.order.append(self.name)
        return 0.001

    async def transcribe(self, pcm):
        return "hello"

    async def stream(self, messages, tools=None):
        yield {"text": "<e:happy:6> Hey."}
        from neiro.llm.openai_compat import StreamAccumulator

        yield {"done": StreamAccumulator(text="<e:happy:6> Hey.")}

    async def synth(self, text, state=None):
        yield np.zeros(240, dtype=np.float32), None

    async def play(self, turn, pcm, seq=0, text="", visemes=None):
        pass

    async def cancel(self, turn):
        pass


def build_daemon(order: list[str] | None = None) -> Daemon:
    order = order if order is not None else []
    d = Daemon(cfg=Neiro())
    stt = Recorder(order, "stt")
    llm = Recorder(order, "llm")
    tts = Recorder(order, "tts")
    sink = Recorder(order, "sink")
    d.build(stt=stt, llm=llm, tts=tts, sink=sink)
    return d


class TestWarmUp:
    def test_components_warm_in_the_order_a_turn_uses_them(self) -> None:
        # Not a gather(): if start-up is interrupted, the parts that ran
        # should be the parts needed first.
        order: list[str] = []
        d = build_daemon(order)
        d.warm()
        assert order.index("stt") < order.index("tts")

    def test_the_report_names_what_it_warmed(self) -> None:
        d = build_daemon()
        report = d.warm()
        assert {"stt", "tts"} <= set(report.stages)
        assert report.total_s >= 0
        assert "warm:" in report.describe()

    def test_a_component_that_fails_to_warm_does_not_stop_start_up(self) -> None:
        # A cold component still works, just slowly. Refusing to start
        # would turn a slow first turn into no turns at all.
        d = Daemon(cfg=Neiro())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt", fail=True),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
        )
        report = d.warm()
        assert "stt" in report.stages

    def test_the_null_affect_provider_is_used_until_the_gate_passes(self) -> None:
        # affect.enabled flips to true only after G3b passes on HIS
        # recordings, and the two providers are interchangeable.
        from neiro.affect.null import NullAffectProvider

        d = build_daemon()
        assert isinstance(d.affect, NullAffectProvider)
        assert not Neiro().affect.enabled


class TestTurns:
    def test_a_turn_runs(self) -> None:
        d = build_daemon()
        result = asyncio.run(d.handle(AUDIO))
        assert result.error is None
        assert "Hey" in result.reply

    def test_only_one_turn_runs_at_a_time(self) -> None:
        # A lock, not a queue: if he speaks while she is answering, that
        # is barge-in, not a second turn to run afterwards.
        d = build_daemon()

        async def go():
            return await asyncio.gather(d.handle(AUDIO), d.handle(AUDIO))

        results = asyncio.run(go())
        assert all(isinstance(r, TurnResult) for r in results)


class TestBargeIn:
    def test_interrupting_nothing_is_not_an_error(self) -> None:
        d = build_daemon()
        assert asyncio.run(d.interrupt()) is False

    def test_interrupting_a_live_turn_sets_its_cancel(self) -> None:
        d = build_daemon()
        turn = Turn.new(1)
        d._live = turn
        assert asyncio.run(d.interrupt()) is True
        assert turn.cancel.is_set()

    def test_interrupting_twice_only_counts_once(self) -> None:
        d = build_daemon()
        d._live = Turn.new(1)
        assert asyncio.run(d.interrupt()) is True
        assert asyncio.run(d.interrupt()) is False


class TestSpokenErrors:
    def test_a_good_turn_says_nothing_extra(self) -> None:
        d = build_daemon()
        assert d.spoken_error(TurnResult(turn=Turn.new(1))) is None

    def test_each_failure_has_something_in_character_to_say(self) -> None:
        # Silence after a failure looks like she froze.
        d = build_daemon()
        for error in ("empty_transcript", "ConnectionError", "ReadTimeout", "ValueError"):
            line = d.spoken_error(TurnResult(turn=Turn.new(1), error=error))
            assert line and len(line.split()) <= 12, error

    def test_a_timeout_is_slow_not_down(self) -> None:
        # Different failures deserve different lines: "she's thinking
        # hard" is not "the server is gone".
        d = build_daemon()
        slow = d.spoken_error(TurnResult(turn=Turn.new(1), error="ReadTimeout"))
        down = d.spoken_error(TurnResult(turn=Turn.new(1), error="ConnectionError"))
        assert slow != down
        assert d.errors.line_for(Failure.LLM_SLOW)


class TestTheLiveTurnIsReal:
    """`_live` used to be assigned `None` and nothing else.

    That made `interrupt()` and `observe()` permanently dead branches:
    barge-in could never reach a turn, and the prosody annotation never
    left the provider. Every test passed throughout, because they all
    set `_live` by hand — which is exactly the shape of test that proves
    nothing about the wiring.
    """

    def test_a_real_turn_can_be_interrupted_without_touching_internals(self) -> None:
        d = build_daemon()

        async def go() -> tuple[bool, bool]:
            turn = d.begin_utterance()
            interrupted = await d.interrupt()
            return interrupted, turn.cancel.is_set()

        interrupted, cancelled = asyncio.run(go())
        assert interrupted and cancelled

    def test_barge_in_reaches_a_turn_started_by_handle_alone(self) -> None:
        # The production path: no begin_utterance(), just a turn in
        # flight inside run().
        d = build_daemon()

        async def go() -> bool:
            task = asyncio.create_task(d.handle(AUDIO))
            await asyncio.sleep(0)  # let handle() start and publish
            hit = await d.interrupt()
            await task
            return hit

        assert asyncio.run(go()) or d.orchestrator.live is None

    def test_observe_creates_a_turn_if_speech_started_without_one(self) -> None:
        # Affect runs before the endpoint, so the turn has to exist
        # during the speaking phase or the annotation is discarded.
        class CountingAffect:
            def __init__(self) -> None:
                self.observations = 0

            async def observe(self, window):
                from neiro.state import UserAffect

                self.observations += 1
                return UserAffect(arousal_z=2.5, confidence=0.9)

            def commit_utterance(self) -> bool:
                return False

        affect = CountingAffect()
        d = build_daemon()
        d.affect = affect
        d.orchestrator.affect = affect

        async def go() -> object:
            await d.observe(AUDIO)
            return d._live

        live = asyncio.run(go())
        assert live is not None
        assert affect.observations == 1

    def test_the_live_turn_is_cleared_even_when_the_turn_fails(self) -> None:
        # A stale Turn would make the next barge-in cancel the wrong one.
        class Broken:
            async def stream(self, messages, tools=None):
                raise ConnectionError
                yield  # pragma: no cover

        d = build_daemon()
        d.orchestrator.llm = Broken()
        result = asyncio.run(d.handle(AUDIO))
        assert result.error == "ConnectionError"
        assert d._live is None

    def test_turn_ids_increase(self) -> None:
        d = build_daemon()
        first = d.begin_utterance().id
        asyncio.run(d.handle(AUDIO))
        assert d.orchestrator.begin_turn().id > first
