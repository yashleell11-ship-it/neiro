"""The daemon: warm-up order, one turn at a time, and barge-in.

Wiring should be boring — if it needs cleverness the Protocols were
wrong — so these tests are mostly about the three things the daemon
genuinely owns and nothing else can.

Nothing here touches a private field. A test that sets `_live` by hand
proves nothing about the wiring it is named for: barge-in was dead code
with every test green, for exactly that reason, and the fix that
followed cancelled every turn that had a speaking phase — also with
every test green.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from neiro.config import Neiro
from neiro.daemon import Daemon
from neiro.orchestrator import TurnResult
from neiro.speech.errors import Failure
from neiro.state import Turn, UserAffect

AUDIO = np.zeros(16000, dtype=np.float32)

# What the affect fakes report: past the dead band and confident, so
# describe() turns it into words rather than omitting it.
HEARD = UserAffect(arousal_z=2.5, confidence=0.9)

# A daemon that never comes back is a hang in production; here it has to
# be a failure, not a stuck suite. Every test that parks a turn runs
# under this deadline.
DEADLINE_S = 2.0


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, DEADLINE_S))


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


class CountingAffect:
    """An affect provider that reports HEARD and counts the windows it saw."""

    def __init__(self) -> None:
        self.observations = 0

    async def observe(self, window) -> UserAffect:
        self.observations += 1
        return HEARD

    def commit_utterance(self) -> bool:
        return False


class Gate:
    """An orchestrator whose `run()` stays in flight until told otherwise.

    Everything the daemon owns — the lock, barge-in, the live turn — is
    about what happens WHILE a turn is running, and a real turn on fake
    providers is over in microseconds, too fast to overlap anything.
    This one parks until the test releases it or barge-in cancels it
    (the real `run()` returns at its next check once `cancel` is set),
    and counts how many turns were in flight at once.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.turns: list[Turn] = []
        self.inflight = 0
        self.peak = 0
        self._turn_id = 0

    def begin_turn(self) -> Turn:
        self._turn_id += 1
        return Turn.new(turn_id=self._turn_id)

    async def observe_while_speaking(self, turn: Turn, window) -> UserAffect:
        turn.user_affect = HEARD
        return HEARD

    async def wait_entered(self, task: asyncio.Task) -> None:
        """Block until `task`'s run() is in flight — or `task` died first,
        in which case its exception is the failure, not a deadline."""
        entered = asyncio.ensure_future(self.entered.wait())
        await asyncio.wait({entered, task}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            entered.cancel()
            task.result()
        self.entered.clear()

    async def run(self, audio, annotation=None, turn=None) -> TurnResult:
        turn = turn if turn is not None else self.begin_turn()
        self.turns.append(turn)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.entered.set()
        released = asyncio.ensure_future(self.release.wait())
        cancelled = asyncio.ensure_future(turn.cancel.wait())
        try:
            await asyncio.wait({released, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            return TurnResult(turn=turn, cancelled=turn.cancel.is_set())
        finally:
            self.inflight -= 1
            for waiter in (released, cancelled):
                waiter.cancel()


def build_daemon(order: list[str] | None = None) -> Daemon:
    order = order if order is not None else []
    d = Daemon(cfg=Neiro())
    stt = Recorder(order, "stt")
    llm = Recorder(order, "llm")
    tts = Recorder(order, "tts")
    sink = Recorder(order, "sink")
    d.build(stt=stt, llm=llm, tts=tts, sink=sink)
    return d


def gated_daemon(gate: Gate) -> Daemon:
    d = Daemon(cfg=Neiro())
    d.orchestrator = gate
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

    def test_the_endpoint_runs_the_turn_speech_start_created(self) -> None:
        # handle() interrupts whatever is in flight before it runs. The
        # turn begin_utterance() prepared is not in flight, it is the
        # one about to run — and cancelling it sent every turn that had
        # a speaking phase back empty and `cancelled`.
        d = build_daemon()
        turn = d.begin_utterance()
        result = asyncio.run(d.handle(AUDIO))
        assert result.turn is turn
        assert not result.cancelled
        assert "Hey" in result.reply

    def test_what_affect_heard_reaches_the_turn_that_runs(self) -> None:
        # Affect runs before the endpoint, so the Turn has to exist
        # during the speaking phase or the reading has nowhere to go:
        # observe() with no turn yet starts one, and handle() runs THAT
        # one rather than a fresh, deaf one.
        affect = CountingAffect()
        d = build_daemon()
        d.affect = affect
        d.orchestrator.affect = affect

        async def go() -> tuple[str | None, TurnResult]:
            annotation = await d.observe(AUDIO)
            return annotation, await d.handle(AUDIO, annotation)

        annotation, result = asyncio.run(go())
        assert affect.observations == 1
        assert annotation is not None
        assert result.turn.user_affect == HEARD
        assert not result.cancelled

    def test_turn_ids_increase(self) -> None:
        d = build_daemon()
        first = d.begin_utterance().id
        asyncio.run(d.handle(AUDIO))
        assert d.orchestrator.begin_turn().id > first


class TestBargeIn:
    def test_interrupting_nothing_is_not_an_error(self) -> None:
        d = build_daemon()
        assert asyncio.run(d.interrupt()) is False

    def test_interrupting_a_live_turn_sets_its_cancel(self) -> None:
        # Driven through handle(): the turn barge-in reaches has to be
        # the one the daemon itself put in flight.
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> tuple[bool, TurnResult]:
            task = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(task)
            return await d.interrupt(), await task

        hit, result = run(go())
        assert hit is True
        assert result.cancelled
        assert gate.turns == [result.turn]

    def test_interrupting_twice_only_counts_once(self) -> None:
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> tuple[bool, bool]:
            task = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(task)
            hits = await d.interrupt(), await d.interrupt()
            await task
            return hits

        assert run(go()) == (True, False)

    def test_his_own_utterance_is_not_something_to_interrupt(self) -> None:
        # Speech-start creates a Turn, but nothing is playing. Barge-in
        # cancels her answer, never the utterance he is in the middle
        # of — which is what one field for both of them used to do.
        d = build_daemon()
        turn = d.begin_utterance()
        assert asyncio.run(d.interrupt()) is False
        assert not turn.cancel.is_set()

    def test_barge_in_cancels_her_answer_not_his_next_utterance(self) -> None:
        # The whole Stage 2 story in one place: he starts speaking while
        # she is answering. Affect observes HIS new utterance, not the
        # turn in flight; the endpoint cancels HER turn and runs his,
        # with the reading still attached.
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> tuple[TurnResult, TurnResult]:
            hers = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(hers)
            annotation = await d.observe(AUDIO)
            his = asyncio.create_task(d.handle(AUDIO, annotation))
            await gate.wait_entered(his)
            gate.release.set()
            return await hers, await his

        hers, his = run(go())
        assert hers.cancelled and not his.cancelled
        assert hers.turn.user_affect is UserAffect.NONE
        assert his.turn.user_affect == HEARD
        assert gate.turns == [hers.turn, his.turn]

    def test_the_next_utterance_cannot_join_a_turn_waiting_for_the_lock(self) -> None:
        # Windows keep arriving while handle() waits for the turn it just
        # cancelled to get out of the way. If it claimed his turn only
        # once it held the lock, the utterance AFTER this one — already
        # being observed — would be what runs on this audio.
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> tuple[TurnResult, TurnResult]:
            hers = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(hers)
            his = asyncio.create_task(d.handle(AUDIO))
            await asyncio.sleep(0)  # his: cancels hers, now parked on the lock
            await d.observe(AUDIO)  # the utterance after his begins
            await gate.wait_entered(his)
            gate.release.set()
            return await hers, await his

        hers, his = run(go())
        assert hers.cancelled and not his.cancelled
        assert his.turn.user_affect is UserAffect.NONE

    def test_a_turn_torn_down_mid_flight_leaves_nothing_to_cancel(self) -> None:
        # The live turn is cleared in a `finally`. The path that needs
        # it is the handle() task itself being cancelled — shutdown —
        # while a turn is parked; a stale Turn would make the next
        # barge-in cancel the wrong one.
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> bool:
            task = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(task)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return await d.interrupt()

        assert run(go()) is False


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
