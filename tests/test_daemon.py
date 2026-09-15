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
import json

import httpx
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from elizabeth.config import Elizabeth
from elizabeth.daemon import Daemon
from elizabeth.llm.ollama_native import OllamaNativeLlm
from elizabeth.llm.openai_compat import StreamAccumulator
from elizabeth.orchestrator import TurnResult
from elizabeth.speech.errors import ErrorSpeech, Failure
from elizabeth.state import Turn, UserAffect
from elizabeth.tools.audit import ATTEMPTED, SUCCEEDED, AuditLog
from elizabeth.tools.builtin import build_registry
from elizabeth.tools.confirm import NotificationConfirmer
from elizabeth.tools.registry import ToolNotConfirmed, ToolRegistry, ToolSpec
from elizabeth.tools.tiers import Tier

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
        from elizabeth.llm.openai_compat import StreamAccumulator

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
    d = Daemon(cfg=Elizabeth())
    stt = Recorder(order, "stt")
    llm = Recorder(order, "llm")
    tts = Recorder(order, "tts")
    sink = Recorder(order, "sink")
    d.build(stt=stt, llm=llm, tts=tts, sink=sink)
    return d


def gated_daemon(gate: Gate) -> Daemon:
    d = Daemon(cfg=Elizabeth())
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
        d = Daemon(cfg=Elizabeth())
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
        from elizabeth.affect.null import NullAffectProvider

        d = build_daemon()
        assert isinstance(d.affect, NullAffectProvider)
        assert not Elizabeth().affect.enabled


class TestTurns:
    def test_a_turn_runs(self) -> None:
        d = build_daemon()
        result = asyncio.run(d.handle(AUDIO))
        assert result.error is None
        assert "Hey" in result.reply

    def test_only_one_turn_runs_at_a_time(self) -> None:
        # A lock, not a queue: if he speaks while she is answering, that
        # is barge-in, not a second turn to run afterwards. Observed as
        # an overlap count — two TurnResults come back whether the turns
        # were serialised or interleaved, so their type proves nothing.
        gate = Gate()
        d = gated_daemon(gate)

        async def go() -> list[TurnResult]:
            hers = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(hers)
            his = asyncio.create_task(d.handle(AUDIO))
            await gate.wait_entered(his)
            gate.release.set()
            return await asyncio.gather(hers, his)

        first, second = run(go())
        assert gate.peak == 1
        assert first.cancelled and not second.cancelled

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

    def test_a_denied_tool_is_not_a_failed_one(self) -> None:
        # He said no. "That didn't work" would claim she tried.
        d = build_daemon()
        denied = d.spoken_error(TurnResult(turn=Turn.new(1), error="ToolNotConfirmed"))
        assert denied == ErrorSpeech().line_for(Failure.TOOL_DENIED)
        # Every other tool failure is hers, and rotates through the
        # TOOL_FAILED lines in step with a fresh speaker.
        expected = ErrorSpeech()
        for error in ("ToolRejected", "RateLimited", "TooManyToolRounds", "OSError"):
            failed = d.spoken_error(TurnResult(turn=Turn.new(1), error=error))
            assert failed == expected.line_for(Failure.TOOL_FAILED), error
            assert failed != denied


# -- tools --------------------------------------------------------------


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PercentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percent: int = Field(ge=0, le=100)


class CallingLlm(Recorder):
    """Asks for `battery` on its first request, answers on the second."""

    def __init__(self) -> None:
        super().__init__([], "llm")
        self.requests: list[tuple[list[dict], list[dict] | None]] = []

    async def stream(self, messages, tools=None):
        self.requests.append((list(messages), tools))
        accumulator = StreamAccumulator()
        if len(self.requests) == 1:
            yield {"text": "<e:neutral:5>"}
            accumulator.add_delta(
                {"tool_calls": [{"index": 0, "function": {"name": "battery", "arguments": "{}"}}]}
            )
        else:
            yield {"text": "<e:happy:7> Ninety six percent."}
        yield {"done": accumulator}


def fake_registry(tmp_path, confirm=None) -> tuple[ToolRegistry, AuditLog, list[int]]:
    ran: list[int] = []
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    r = ToolRegistry(confirm=confirm, audit=audit)
    r.register(
        ToolSpec(
            name="battery",
            description="How much charge is left.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=lambda: ran.append(1) or "Ninety six percent, charging.",
        )
    )
    r.register(
        ToolSpec(
            name="volume",
            description="Set the speaker volume.",
            tier=Tier.YELLOW,
            args_model=PercentArgs,
            handler=lambda percent: f"Volume {percent}.",
        )
    )
    return r, audit, ran


class FakeNotify:
    """A `notify-send --wait` that prints the chosen action and records
    the question it was asked."""

    def __init__(self, chosen: str) -> None:
        self.chosen = chosen
        self.questions: list[str] = []

    def __call__(self, cmd, **kw):
        self.questions.append(cmd[-1])
        return type("R", (), {"stdout": self.chosen})()


class TestTools:
    def test_build_registers_the_builtin_tools_and_hands_them_over(self) -> None:
        # The orchestrator only ever sees a registry; this is the one
        # place that decides what is in it.
        d = build_daemon()
        assert "system_stats" in d.tools and "set_volume" in d.tools
        assert d.orchestrator.tools is d.tools
        offered = [t["function"]["name"] for t in d.orchestrator.tool_schemas()]
        assert "system_stats" in offered
        # YELLOW tools are offered because the notification can ask.
        assert "set_volume" in offered

    def test_an_injected_registry_is_the_one_used(self, tmp_path) -> None:
        r, _, _ = fake_registry(tmp_path)
        d = Daemon(cfg=Elizabeth())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            tools=r,
        )
        assert d.tools is r and d.orchestrator.tools is r
        assert [t["function"]["name"] for t in d.orchestrator.tool_schemas()] == ["battery"]

    def test_an_injected_empty_registry_stays_empty(self) -> None:
        # A registry is sized, so an empty one is falsy: `tools or
        # builtins` handed the orchestrator every builtin tool when the
        # caller had asked for none.
        empty = ToolRegistry()
        d = Daemon(cfg=Elizabeth())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            tools=empty,
        )
        assert d.tools is empty and d.orchestrator.tools is empty
        assert d.orchestrator.tool_schemas() == []

    def test_confirm_none_is_the_notification_gate_not_no_gate(self) -> None:
        # `confirm=None` is the absence of an override: YELLOW tools are
        # offered and the notification asks. GREEN-only is the registry's
        # own property, decided where it is built and handed in whole.
        order: list[str] = []
        asks = Daemon(cfg=Elizabeth())
        asks.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            confirm=None,
        )
        assert asks.tools.can_confirm
        assert "set_volume" in [t["function"]["name"] for t in asks.orchestrator.tool_schemas()]

        quiet = Daemon(cfg=Elizabeth())
        quiet.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            tools=build_registry(confirm=None),
        )
        assert not quiet.tools.can_confirm
        offered = [t["function"]["name"] for t in quiet.orchestrator.tool_schemas()]
        assert "system_stats" in offered and "set_volume" not in offered

    def test_a_green_tool_runs_through_handle(self, tmp_path) -> None:
        # The whole Stage 3 loop from the daemon's door: he asks, the
        # model calls, the tool runs, the result comes back, she answers.
        r, audit, ran = fake_registry(tmp_path)
        llm = CallingLlm()
        d = Daemon(cfg=Elizabeth())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=llm,
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            tools=r,
        )
        result = run(d.handle(AUDIO))
        assert result.error is None and not result.cancelled
        assert ran == [1]
        assert result.reply == "Ninety six percent."
        assert len(llm.requests) == 2
        assert llm.requests[1][0][-1]["role"] == "tool"
        assert [e["event"] for e in audit.entries] == [ATTEMPTED, SUCCEEDED]
        assert audit.entries[0]["turn"] == result.turn.id

    def test_a_tool_turn_completes_on_the_shipped_backend(self, tmp_path) -> None:
        # `build()`'s default LLM is the ollama-native client, and ollama
        # 0.33.2 answers HTTP 400 to the OpenAI-shaped assistant message
        # the orchestrator remembers — arguments as a JSON string. With
        # fakes on both sides every test above passed while every real
        # tool turn ended in error after the tool had already run. This
        # transport plays ollama, including that rejection.
        r, _, ran = fake_registry(tmp_path)
        bodies: list[dict] = []

        def ollama(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            for message in body["messages"]:
                for call in message.get("tool_calls") or []:
                    if not isinstance(call["function"]["arguments"], dict):
                        error = "Value looks like object, but can't find closing '}' symbol"
                        return httpx.Response(400, json={"error": error})
            if len(bodies) == 1:
                calls = [{"function": {"name": "battery", "arguments": {}}}]
                message = {"role": "assistant", "content": "", "tool_calls": calls}
            else:
                message = {"role": "assistant", "content": "<e:happy:7> Ninety six percent."}
            lines = [
                {"message": message, "done": False},
                {"message": {"role": "assistant", "content": ""}, "done": True},
            ]
            return httpx.Response(200, content="".join(json.dumps(l) + "\n" for l in lines))

        d = Daemon(cfg=Elizabeth())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=OllamaNativeLlm(
                base_url="http://ollama.test", transport=httpx.MockTransport(ollama)
            ),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            tools=r,
        )
        result = run(d.handle(AUDIO))
        assert result.error is None, result.error
        assert ran == [1] and result.reply == "Ninety six percent."
        assert len(bodies) == 2
        assert [m["role"] for m in bodies[1]["messages"]] == ["system", "user", "assistant", "tool"]

    def test_the_gate_and_log_handed_in_reach_the_real_registry(self, tmp_path) -> None:
        # `confirm` and `audit` replace just those two parts of the
        # builtin registry, so a test can deny everything and log to a
        # temporary file while the tools themselves stay real.
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        d = Daemon(cfg=Elizabeth())
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            confirm=lambda *_: False,
            audit=audit,
        )
        with pytest.raises(ToolNotConfirmed):
            d.tools.call("set_volume", {"percent": 30}, turn_id=1)
        assert audit.entries and audit.entries[0]["tool"] == "set_volume"


class TestConfirmation:
    """The default gate for `elizabeth talk`: a clickable notification."""

    def _ask(self, chosen: str, tool: str = "set_volume", args=None) -> tuple[bool, FakeNotify]:
        notify = FakeNotify(chosen)
        d = build_daemon()
        d.notifier = NotificationConfirmer(runner=notify)
        spec, parsed = d.tools.validate(tool, args if args is not None else {"percent": 30})
        return d.notification_confirm(spec, parsed, "nonce"), notify

    def test_a_clicked_yes_allows(self) -> None:
        allowed, _ = self._ask("yes\n")
        assert allowed

    def test_a_clicked_no_denies(self) -> None:
        allowed, _ = self._ask("no\n")
        assert not allowed

    def test_a_dismissed_notification_denies(self) -> None:
        # Silence is no. An assistant that acts on a dismissed popup is
        # worse than one that asks twice.
        allowed, _ = self._ask("")
        assert not allowed

    def test_the_question_says_what_would_run(self) -> None:
        # A yes must be a yes to THIS: the tool and every argument.
        _, notify = self._ask("yes\n")
        assert notify.questions == ["set volume, percent 30?"]
        _, notify = self._ask("yes\n", "adjust_brightness", {"direction": "down", "steps": 2})
        assert notify.questions == ["adjust brightness, direction down, steps 2?"]

    def test_the_default_registry_asks_through_the_notifier(self, tmp_path, monkeypatch) -> None:
        # Not just that the method works — that the registry the daemon
        # built actually calls it. A registry built with `confirm=None`
        # refuses every YELLOW tool with the same exception, so this
        # checks the yes path: the notifier is asked and the tool runs.
        from elizabeth.tools import system

        notify = FakeNotify("yes\n")
        d = Daemon(cfg=Elizabeth())
        d.notifier = NotificationConfirmer(runner=notify)
        order: list[str] = []
        d.build(
            stt=Recorder(order, "stt"),
            llm=Recorder(order, "llm"),
            tts=Recorder(order, "tts"),
            sink=Recorder(order, "sink"),
            audit=AuditLog(path=tmp_path / "audit.jsonl"),
        )
        # The handler's subprocess seam is stubbed so the suite never
        # actually mutes the speakers.
        monkeypatch.setattr(system, "_run", lambda cmd, timeout=0.0: "")
        d.tools.call("set_mute", {"state": "toggle"}, turn_id=1)
        assert notify.questions == ["set mute, state toggle?"]
