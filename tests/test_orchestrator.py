"""The turn loop, exercised entirely with fakes.

No mic, no model, no browser — which is the point: the whole pipeline's
control flow (backpressure, cancellation, history, the timeline) is
logic, and logic that can only be tested with hardware never gets
tested.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from elizabeth.llm.openai_compat import StreamAccumulator
from elizabeth.orchestrator import CHUNK_QUEUE_DEPTH, MAX_TOOL_ROUNDS, Orchestrator, TurnResult
from elizabeth.state import NEUTRAL_STATE, ElizabethState, EmotionLabel, UserAffect
from elizabeth.tools.audit import DENIED, FAILED, AuditLog
from elizabeth.tools.registry import ToolRegistry, ToolSpec
from elizabeth.tools.tiers import Tier

AUDIO = np.zeros(16000, dtype=np.float32)


class FakeStt:
    def __init__(self, text: str = "what's my battery at") -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, pcm: np.ndarray) -> str:
        self.calls += 1
        return self.text


class FakeLlm:
    """Streams a reply in small fragments, like a real token stream."""

    def __init__(self, reply: str = "<e:happy:7> Ninety six percent. Still charging.") -> None:
        self.reply = reply
        self.seen_messages: list[list[dict]] = []
        self.delay = 0.0

    async def stream(self, messages: list[dict], tools=None) -> AsyncIterator[dict]:
        self.seen_messages.append(messages)
        for i in range(0, len(self.reply), 5):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield {"text": self.reply[i : i + 5]}
        yield {"done": StreamAccumulator(text=self.reply)}


class FakeTts:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.states: list = []

    async def synth(self, text: str, state=None):
        self.texts.append(text)
        self.states.append(state)
        yield np.zeros(240, dtype=np.float32), None


class FakeSink:
    def __init__(self) -> None:
        self.played: list[tuple[int, str]] = []
        self.cancelled = 0

    async def play(self, turn, pcm, seq=0, text="", visemes=None) -> None:
        self.played.append((seq, text))

    async def cancel(self, turn) -> None:
        self.cancelled += 1


class FakeAffect:
    def __init__(self) -> None:
        self.observations = 0
        self.commits = 0
        self.discards = 0

    async def observe(self, window: np.ndarray) -> UserAffect:
        self.observations += 1
        return UserAffect(arousal_z=2.5, confidence=0.9)

    def commit_utterance(self) -> bool:
        self.commits += 1
        return False

    def discard_utterance(self) -> None:
        self.discards += 1


def build(**kw) -> tuple[Orchestrator, dict]:
    parts = {
        "stt": kw.get("stt") or FakeStt(),
        "llm": kw.get("llm") or FakeLlm(),
        "tts": kw.get("tts") or FakeTts(),
        "sink": kw.get("sink") or FakeSink(),
        "affect": kw.get("affect"),
    }
    return Orchestrator(**parts), parts


class TestHappyPath:
    def test_a_turn_runs_end_to_end(self) -> None:
        orch, parts = build()
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None and not result.cancelled
        assert result.transcript == "what's my battery at"
        assert "Ninety six percent" in result.reply
        assert parts["sink"].played

    def test_the_emotion_tag_never_reaches_the_voice(self) -> None:
        # A TTS reading "<e:happy:7>" aloud is a specific, embarrassing
        # failure, and the whole reason the tag is stripped upstream.
        orch, parts = build()
        asyncio.run(orch.run(AUDIO))
        for text in parts["tts"].texts:
            assert "<e:" not in text
        assert "<e:" not in "".join(t for _, t in parts["sink"].played)

    def test_her_state_reaches_the_synthesiser(self) -> None:
        orch, parts = build()
        result = asyncio.run(orch.run(AUDIO))
        assert result.state.label is EmotionLabel.HAPPY
        assert any(s is not None and s.label is EmotionLabel.HAPPY for s in parts["tts"].states)

    def test_the_timeline_records_the_measured_boundaries(self) -> None:
        # metrics.py turns this dict into the one number: endpoint ->
        # first audio. Missing stamps make the metric silently wrong.
        orch, _ = build()
        result = asyncio.run(orch.run(AUDIO))
        for stamp in ("endpoint", "stt_start", "stt_done", "tts_first_chunk", "sink_first_sent"):
            assert stamp in result.turn.timeline, stamp
        assert result.turn.timeline["stt_done"] >= result.turn.timeline["stt_start"]

    def test_sequence_zero_is_the_first_thing_played(self) -> None:
        orch, parts = build()
        asyncio.run(orch.run(AUDIO))
        assert parts["sink"].played[0][0] == 0


class TestPromptAndHistory:
    def test_the_system_prompt_is_first_and_identical_every_turn(self) -> None:
        # The KV prefix cache depends on it byte-for-byte. A drift is a
        # 0.3-1.3 s latency cliff with no error message.
        orch, parts = build()
        asyncio.run(orch.run(AUDIO))
        asyncio.run(orch.run(AUDIO))
        first, second = parts["llm"].seen_messages
        assert first[0]["role"] == "system"
        assert first[0] == second[0]

    def test_history_is_appended_never_rewritten(self) -> None:
        orch, parts = build()
        asyncio.run(orch.run(AUDIO))
        asyncio.run(orch.run(AUDIO))
        first, second = parts["llm"].seen_messages
        assert second[: len(first)] == first[: len(first)]
        assert len(second) > len(first)

    def test_history_trims_in_pairs(self) -> None:
        # Dropping a lone user turn leaves an assistant reply answering
        # nothing, which reads as her having hallucinated the question.
        orch, parts = build()
        orch._history_limit = 2
        for _ in range(6):
            asyncio.run(orch.run(AUDIO))
        history = parts["llm"].seen_messages[-1][1:-1]
        assert len(history) % 2 == 0
        assert all(history[i]["role"] == "user" for i in range(0, len(history), 2))

    def test_the_annotation_goes_on_the_user_turn_only(self) -> None:
        orch, parts = build()
        asyncio.run(orch.run(AUDIO, annotation="more energy than usual"))
        messages = parts["llm"].seen_messages[0]
        assert "[voice:" in messages[-1]["content"]
        assert "more energy than usual" not in messages[0]["content"]

    def test_forget_clears_it(self) -> None:
        orch, parts = build()
        asyncio.run(orch.run(AUDIO))
        orch.forget()
        asyncio.run(orch.run(AUDIO))
        assert len(parts["llm"].seen_messages[-1]) == 2  # system + this turn


class TestFailures:
    def test_an_empty_transcript_does_not_become_a_reply(self) -> None:
        # The STT confidence floor already rejected it; answering anyway
        # is how an assistant replies to silence.
        orch, parts = build(stt=FakeStt(""))
        result = asyncio.run(orch.run(AUDIO))
        assert result.error == "empty_transcript"
        assert not parts["sink"].played

    def test_a_whitespace_transcript_counts_as_empty(self) -> None:
        orch, _ = build(stt=FakeStt("   \n "))
        assert asyncio.run(orch.run(AUDIO)).error == "empty_transcript"

    def test_a_failing_llm_does_not_raise_out_of_the_turn(self) -> None:
        # She has to say something in character about it, which she
        # cannot do if the turn threw.
        class Broken:
            async def stream(self, messages, tools=None):
                raise ConnectionError("llama-server is not running")
                yield  # pragma: no cover

        result = asyncio.run(build(llm=Broken())[0].run(AUDIO))
        assert result.error == "ConnectionError"
        assert not result.cancelled

    def test_a_failing_turn_is_not_remembered(self) -> None:
        class Broken:
            async def stream(self, messages, tools=None):
                raise ConnectionError
                yield  # pragma: no cover

        orch, _ = build(llm=Broken())
        asyncio.run(orch.run(AUDIO))
        assert orch._history == []


class TestCancellation:
    def test_a_cancelled_turn_is_clipped_from_history(self) -> None:
        # Remembering words she was interrupted before saying makes her
        # refer back to things he never heard.
        class CancellingSink(FakeSink):
            """Fires barge-in the instant the first chunk is played."""

            async def play(self, turn, pcm, seq=0, text="", visemes=None):
                turn.cancel.set()
                await super().play(turn, pcm, seq=seq, text=text, visemes=visemes)

        sink = CancellingSink()
        orch, _ = build(sink=sink)
        result = asyncio.run(orch.run(AUDIO))
        assert result.cancelled
        assert sink.cancelled == 1
        assert orch._history == []

    def test_affect_is_not_committed_on_a_cancelled_turn(self) -> None:
        # An interrupted utterance is not a sample of how he normally
        # sounds.
        affect = FakeAffect()

        class CancellingSink(FakeSink):
            async def play(self, turn, pcm, seq=0, text="", visemes=None):
                turn.cancel.set()

        orch, _ = build(affect=affect, sink=CancellingSink())
        result = asyncio.run(orch.run(AUDIO))
        assert result.cancelled
        assert affect.commits == 0
        # ...but it is discarded, not left staged for the next turn.
        assert affect.discards == 1


class TestAffect:
    def test_observing_happens_during_speech_and_costs_the_turn_nothing(self) -> None:
        affect = FakeAffect()
        orch, _ = build(affect=affect)

        async def go():
            from elizabeth.state import Turn

            turn = Turn.new(0)
            for _ in range(4):
                await orch.observe_while_speaking(turn, AUDIO)
            return turn

        turn = asyncio.run(go())
        assert affect.observations == 4
        assert turn.user_affect.confidence == 0.9

    def test_the_baseline_is_updated_once_per_utterance(self) -> None:
        # Not once per analysis window, or one long sentence redefines
        # his normal.
        affect = FakeAffect()
        orch, _ = build(affect=affect)
        asyncio.run(orch.run(AUDIO))
        assert affect.commits == 1

    def test_a_rejected_utterance_is_discarded_not_committed(self) -> None:
        # The STT floor blanking the transcript does not un-hear the
        # windows affect observed while he spoke. The provider staged
        # them waiting for exactly one close per utterance; the early
        # return skipped it, and the next turn's band was decided against
        # this utterance's history. The close is a DISCARD: nothing
        # usable was said, so a possibly hallucinated window must not
        # become part of his normal.
        affect = FakeAffect()
        orch, _ = build(stt=FakeStt(""), affect=affect)

        async def go() -> TurnResult:
            turn = orch.begin_turn()
            await orch.observe_while_speaking(turn, AUDIO)
            return await orch.run(AUDIO, turn=turn)

        result = asyncio.run(go())
        assert result.error == "empty_transcript"
        assert affect.observations == 1
        assert affect.commits == 0
        assert affect.discards == 1

    def test_a_downstream_failure_still_commits_what_he_said(self) -> None:
        # LLM down is her problem, not a fact about his voice.
        class Broken:
            async def stream(self, messages, tools=None):
                raise ConnectionError("llama-server is not running")
                yield  # pragma: no cover

        affect = FakeAffect()
        orch, _ = build(affect=affect, llm=Broken())
        result = asyncio.run(orch.run(AUDIO))
        assert result.error == "ConnectionError"
        assert affect.commits == 1
        assert affect.discards == 0

    def test_no_affect_provider_is_fine(self) -> None:
        orch, _ = build(affect=None)
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert result.turn.user_affect == UserAffect.NONE


class TestBargeInOnAFullQueue:
    """The case that deadlocked the assistant permanently.

    `produce()`'s `finally` does `await queue.put(None)`. When
    cancellation reached the producer while it was parked on a FULL
    queue — the designed steady state, since the queue being full IS the
    backpressure — that cleanup started a *fresh* put on a queue nobody
    would drain again. asyncio does not re-deliver cancellation to a
    task suspended inside its own finally, so the producer parked
    forever and `run()` never returned. In production `Daemon.handle()`
    holds its lock across that call: the assistant was dead until
    restart, on the barge-in path the module exists for.

    The older cancellation tests miss it because they fire barge-in from
    `sink.play()` on seq 0, immediately after a `queue.get()` freed a
    slot, with a single-chunk FakeTts.
    """

    class MultiChunkTts(FakeTts):
        """Two chunks per sentence, with an await between them, so cancel
        can land mid-sentence — which is what a real TTS does.
        """

        async def synth(self, text: str, state=None):
            self.texts.append(text)
            self.states.append(state)
            for _ in range(2):
                await asyncio.sleep(0.02)
                yield np.zeros(240, dtype=np.float32), None

    class CancelOnFirstChunk(FakeSink):
        async def play(self, turn, pcm, seq=0, text="", visemes=None) -> None:
            turn.cancel.set()
            await super().play(turn, pcm, seq=seq, text=text, visemes=visemes)

    def test_it_does_not_deadlock(self) -> None:
        long_reply = "<e:neutral:5> " + " ".join(f"Sentence number {i}." for i in range(12))

        async def go() -> TurnResult:
            orch = Orchestrator(
                stt=FakeStt(),
                llm=FakeLlm(long_reply),
                tts=self.MultiChunkTts(),
                sink=self.CancelOnFirstChunk(),
            )
            # Without the drain this never returns.
            return await asyncio.wait_for(orch.run(AUDIO), timeout=5)

        result = asyncio.run(go())
        assert result.cancelled

    def test_the_interrupted_reply_is_still_clipped_from_history(self) -> None:
        # Fixing the hang must not resurrect words she never said.
        long_reply = "<e:neutral:5> " + " ".join(f"Sentence number {i}." for i in range(12))

        async def go() -> Orchestrator:
            orch = Orchestrator(
                stt=FakeStt(),
                llm=FakeLlm(long_reply),
                tts=self.MultiChunkTts(),
                sink=self.CancelOnFirstChunk(),
            )
            await asyncio.wait_for(orch.run(AUDIO), timeout=5)
            return orch

        assert asyncio.run(go())._history == []


class TestBargeInBeforeTheFirstToken:
    """A barge-in while she is still thinking.

    Between the STT result and the LLM's first token the consumer is
    parked on the sentence queue with nothing coming. A cancel check
    placed after `queue.get()` only runs once an item arrives, so an
    interruption in that window waited for the model he had just
    interrupted to start talking — bounded by the provider's HTTP
    timeout, 120 s for Ollama — with `Daemon.handle()`'s lock held the
    whole time, so his new utterance could not start either.

    The older cancellation tests all fire barge-in from `sink.play()`,
    i.e. only once tokens are already flowing.
    """

    class StalledLlm:
        """Never produces a token: a wedged llama-server, or a cold
        prefill that is still going when he interrupts.
        """

        def __init__(self) -> None:
            self.asked = asyncio.Event()
            self.torn_down = False

        async def stream(self, messages, tools=None):
            self.asked.set()
            try:
                await asyncio.Event().wait()  # nobody will ever set it
                yield {"text": ""}  # pragma: no cover
            finally:
                self.torn_down = True

    def test_cancel_wakes_a_consumer_parked_on_an_empty_queue(self) -> None:
        llm = self.StalledLlm()
        orch, parts = build(llm=llm)

        async def go() -> TurnResult:
            turn = orch.begin_turn()
            task = asyncio.create_task(orch.run(AUDIO, turn=turn))
            # Once the LLM has been asked, the consumer is already parked
            # on `get()` — it got there before the producer even ran.
            await llm.asked.wait()
            assert not task.done()
            turn.cancel.set()
            # Without a cancel-aware wait this is where it sat until the
            # model spoke, which this model never does.
            return await asyncio.wait_for(task, timeout=1)

        result = asyncio.run(go())
        assert result.cancelled
        assert result.reply == ""
        assert parts["sink"].cancelled == 1
        assert orch._history == []
        # The producer was awaited, not abandoned with its HTTP
        # connection open.
        assert llm.torn_down

    def test_a_barge_in_during_stt_never_asks_the_llm(self) -> None:
        # The stage checks before its await; it has to check after it
        # too, or the request goes out and is cancelled a token later.
        orch, parts = build()
        turn = orch.begin_turn()

        class InterruptedStt(FakeStt):
            async def transcribe(self, pcm):
                turn.cancel.set()
                return await super().transcribe(pcm)

        orch.stt = InterruptedStt()
        result = asyncio.run(orch.run(AUDIO, turn=turn))
        assert result.cancelled
        assert parts["llm"].seen_messages == []


class TestBackpressure:
    """The producer blocks when the voice is behind.

    Two earlier versions of this test could not fail on the property
    they were named for: one asserted `CHUNK_QUEUE_DEPTH <= 4`, a
    constant, and its replacement built its own `asyncio.Queue` and
    proved that asyncio's queue blocks. Dropping `maxsize` from the
    orchestrator's queue — no backpressure at all, the LLM running the
    whole reply ahead of the voice — left both green. This one stalls
    the synthesiser and watches how far the LLM gets.
    """

    SENTENCES = 12

    class GatedTts(FakeTts):
        """Holds the first sentence until released: the voice is behind."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def synth(self, text: str, state=None):
            self.texts.append(text)
            self.states.append(state)
            await self.gate.wait()
            yield np.zeros(240, dtype=np.float32), None

    class OneSentencePerEventLlm:
        """Streams one complete sentence per event and counts how many it
        has handed over, so the test can see exactly how far ahead of
        the voice the model was allowed to run.
        """

        def __init__(self, sentences: int) -> None:
            self.sentences = sentences
            self.streamed = 0
            self.finished = False

        async def stream(self, messages: list[dict], tools=None) -> AsyncIterator[dict]:
            yield {"text": "<e:neutral:5> "}
            for i in range(self.sentences):
                self.streamed += 1
                yield {"text": f"Sentence number {i}. "}
            self.finished = True
            yield {"done": StreamAccumulator(text="")}

    def test_the_producer_blocks_when_the_voice_stalls(self) -> None:
        llm = self.OneSentencePerEventLlm(self.SENTENCES)
        tts = self.GatedTts()
        orch, _ = build(llm=llm, tts=tts)

        async def go() -> TurnResult:
            task = asyncio.create_task(orch.run(AUDIO))
            # Nothing in the fakes sleeps, so the pipeline reaches a
            # standstill — voice parked on the gate, producer parked on
            # `put()` — within a handful of loop iterations. Spinning
            # is deterministic where a timed sleep would not be.
            for _ in range(50):
                await asyncio.sleep(0)
            assert not task.done()

            # One sentence in the synthesiser, CHUNK_QUEUE_DEPTH waiting
            # behind it, and one more blocked in `put()`. Without the
            # bound the model streams all twelve and finishes.
            assert llm.streamed <= CHUNK_QUEUE_DEPTH + 2, llm.streamed
            assert not llm.finished
            assert tts.texts == ["Sentence number 0."]

            tts.gate.set()
            return await asyncio.wait_for(task, timeout=5)

        result = asyncio.run(go())
        assert result.error is None and not result.cancelled
        assert llm.finished
        # Releasing the voice loses nothing and reorders nothing.
        assert tts.texts == [f"Sentence number {i}." for i in range(self.SENTENCES)]

    def test_the_lookahead_is_small(self) -> None:
        # The test above proves the bound exists; this one guards its
        # size. Two sentences of lookahead is what a barge-in throws
        # away.
        assert CHUNK_QUEUE_DEPTH <= 4

    def test_a_slow_voice_does_not_lose_sentences(self) -> None:
        class SlowTts(FakeTts):
            async def synth(self, text, state=None):
                await asyncio.sleep(0.005)
                async for chunk in super().synth(text, state):
                    yield chunk

        long_reply = "<e:neutral:5> " + " ".join(f"Sentence number {i}." for i in range(8))
        orch, parts = build(llm=FakeLlm(long_reply), tts=SlowTts())
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert len(parts["tts"].texts) >= 8
        assert "Sentence number 7" in " ".join(parts["tts"].texts)


class TestStateSeparation:
    def test_user_affect_and_her_state_stay_apart(self) -> None:
        # CLAUDE.md rule 5. Sharing one variable is how an assistant ends
        # up reading its own TTS back as the user's mood.
        #
        # Drives both writers: observe_while_speaking() is the only thing
        # that sets user_affect, and run() the only thing that sets
        # elizabeth_state. The previous version ran only run() and inspected
        # the Turn's untouched defaults, so `elizabeth_state = user_affect`
        # inside observe_while_speaking() left the suite green.
        affect = FakeAffect()
        orch, _ = build(affect=affect)

        async def go() -> tuple[TurnResult, UserAffect, ElizabethState]:
            turn = orch.begin_turn()
            for _ in range(3):
                await orch.observe_while_speaking(turn, AUDIO)
            # Snapshot what the speaking phase wrote, before the reply
            # can overwrite either side.
            heard, felt = turn.user_affect, turn.elizabeth_state
            return await orch.run(AUDIO, turn=turn), heard, felt

        result, heard, felt = asyncio.run(go())
        turn = result.turn

        # Observing his voice wrote what was heard and nothing else.
        assert type(heard) is UserAffect
        assert heard.arousal_z == 2.5 and heard.confidence == 0.9
        assert felt is NEUTRAL_STATE

        # The reply wrote what she feels and nothing else.
        assert type(turn.elizabeth_state) is ElizabethState
        assert turn.elizabeth_state.label is EmotionLabel.HAPPY
        assert turn.user_affect is heard

        # The rule itself: two objects, never the same one.
        assert turn.elizabeth_state is not turn.user_affect


# -- tools ---------------------------------------------------------------


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PercentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percent: int = Field(ge=0, le=100)


def call(name: str, *fragments: str) -> tuple:
    """A scripted tool call whose arguments arrive as these JSON
    fragments, one delta each — the shape a real stream has."""
    return ("call", name, fragments or ("{}",))


class ToolLlm:
    """One script per request. A `str` item is a text fragment; a
    `call(...)` item is a tool call, accumulated exactly as the real
    clients do it and surfaced on the `done` event.

    Synchronous by default — no await between events — so a whole
    request is produced before the voice gets to run. `pace` adds a
    sleep between events for the tests that need the voice to keep up.
    """

    def __init__(self, *scripts: list, pace: float = 0.0) -> None:
        self.scripts = list(scripts)
        self.pace = pace
        self.requests: list[tuple[list[dict], list[dict] | None]] = []

    async def stream(self, messages: list[dict], tools=None) -> AsyncIterator[dict]:
        self.requests.append((list(messages), tools))
        script = self.scripts.pop(0) if self.scripts else ["<e:neutral:5> Done."]
        accumulator = StreamAccumulator()
        index = 0
        for item in script:
            if self.pace:
                await asyncio.sleep(self.pace)
            if isinstance(item, str):
                accumulator.add_delta({"content": item})
                yield {"text": item}
                continue
            _, name, fragments = item
            accumulator.add_delta(
                {"tool_calls": [{"index": index, "function": {"name": name, "arguments": ""}}]}
            )
            for fragment in fragments:
                accumulator.add_delta(
                    {"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}
                )
            index += 1
        yield {"done": accumulator}


class Handlers:
    """The tool bodies, recording what they were given."""

    def __init__(self) -> None:
        self.battery_calls = 0
        self.volumes: list[int] = []
        self.seen_state: list[EmotionLabel] = []
        self.turn: object = None
        self.fail: Exception | None = None
        # A blocking handler stands in for a six-second confirmation
        # window. Bounded, so an implementation that blocks the event
        # loop on it fails its test instead of hanging the suite.
        self.block: threading.Event | None = None
        self.block_timeout_s = 2.0
        self.started = threading.Event()
        self.finished = threading.Event()

    def battery(self) -> str:
        self.battery_calls += 1
        if self.turn is not None:
            self.seen_state.append(self.turn.elizabeth_state.label)
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=self.block_timeout_s)
        self.finished.set()
        if self.fail is not None:
            raise self.fail
        return "Ninety six percent, charging."

    def volume(self, percent: int) -> str:
        self.volumes.append(percent)
        return f"Volume {percent} percent."


def registry(
    handlers: Handlers, confirm=None, tmp_path: Path | None = None
) -> tuple[ToolRegistry, AuditLog | None]:
    audit = AuditLog(path=tmp_path / "audit.jsonl") if tmp_path is not None else None
    r = ToolRegistry(confirm=confirm, audit=audit)
    r.register(
        ToolSpec(
            name="battery",
            description="How much charge is left.",
            tier=Tier.GREEN,
            args_model=NoArgs,
            handler=handlers.battery,
        )
    )
    r.register(
        ToolSpec(
            name="volume",
            description="Set the speaker volume.",
            tier=Tier.YELLOW,
            args_model=PercentArgs,
            handler=handlers.volume,
        )
    )
    return r, audit


def build_with_tools(llm: ToolLlm, tools: ToolRegistry | None, **kw) -> tuple[Orchestrator, dict]:
    parts = {
        "stt": FakeStt(),
        "llm": llm,
        "tts": kw.get("tts") or FakeTts(),
        "sink": kw.get("sink") or FakeSink(),
    }
    return Orchestrator(**parts, tools=tools), parts


class TestTools:
    """Tools reachable by voice: the model asks, the registry runs, the
    result goes back, and what she finally says is the second stream.
    """

    def test_a_green_call_runs_and_its_result_feeds_the_second_request(self) -> None:
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("battery")],
            ["<e:happy:7> Ninety six percent. Still charging."],
        )
        orch, parts = build_with_tools(llm, registry(handlers)[0])
        result = asyncio.run(orch.run(AUDIO))

        assert result.error is None and not result.cancelled
        assert handlers.battery_calls == 1
        assert len(llm.requests) == 2
        # The second request is the first plus the call and its result,
        # in the exact OpenAI shape, bound by id.
        first, second = (m for m, _ in llm.requests)
        assert second[: len(first)] == first
        asked, answered = second[len(first) :]
        assert asked["role"] == "assistant" and asked["tool_calls"][0]["type"] == "function"
        assert asked["tool_calls"][0]["function"] == {"name": "battery", "arguments": "{}"}
        assert answered == {
            "role": "tool",
            "tool_call_id": asked["tool_calls"][0]["id"],
            "content": "Ninety six percent, charging.",
        }
        # What she said is the second stream, and only that.
        assert result.reply == "Ninety six percent. Still charging."
        assert parts["tts"].texts == ["Ninety six percent.", "Still charging."]
        assert result.turn.tool_calls[0].name == "battery"

    def test_the_whole_exchange_is_remembered_in_order(self) -> None:
        handlers = Handlers()
        llm = ToolLlm(["<e:neutral:5> ", call("battery")], ["<e:happy:7> Ninety six percent."])
        orch, _ = build_with_tools(llm, registry(handlers)[0])
        asyncio.run(orch.run(AUDIO))
        asyncio.run(orch.run(AUDIO))
        # The next turn sees user, the call, the result, the reply — then
        # its own user turn. Nothing rewritten, nothing missing.
        roles = [m["role"] for m in llm.requests[-1][0]]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]

    def test_the_first_tag_drives_the_face_and_the_second_the_voice(self) -> None:
        # The action-turn contract: the tag on the tool-call stream is
        # what her face shows while the tool runs; the tag on the stream
        # after the result is what her voice carries.
        handlers = Handlers()
        llm = ToolLlm(["<e:relaxed:3> ", call("battery")], ["<e:surprised:8> Only twelve percent!"])
        orch, parts = build_with_tools(llm, registry(handlers)[0])
        turn = orch.begin_turn()
        handlers.turn = turn
        result = asyncio.run(orch.run(AUDIO, turn=turn))

        assert handlers.seen_state == [EmotionLabel.RELAXED]
        assert [s.label for s in parts["tts"].states] == [EmotionLabel.SURPRISED]
        assert result.state.label is EmotionLabel.SURPRISED
        assert turn.elizabeth_state.label is EmotionLabel.SURPRISED
        assert "emotion_resolved" in turn.timeline

    def test_narration_alongside_a_call_is_not_spoken_or_remembered(self) -> None:
        # The prompt forbids "let me check". When the model does it
        # anyway, what has not reached the voice by the time the call
        # appears is dropped, and none of it is her reply.
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", "Checking now. ", call("battery")],
            ["<e:happy:7> Ninety six percent."],
        )
        orch, parts = build_with_tools(llm, registry(handlers)[0])
        result = asyncio.run(orch.run(AUDIO))

        assert result.reply == "Ninety six percent."
        assert parts["tts"].texts == ["Ninety six percent."]
        assert not any("Checking" in json.dumps(m) for m in orch._history)
        assert orch._history[1]["content"] == ""

    def test_arguments_split_across_chunks_arrive_whole(self) -> None:
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("volume", '{"per', 'cent"', ": 4", "0}")],
            ["<e:relaxed:4> Forty it is."],
        )
        orch, _ = build_with_tools(llm, registry(handlers, confirm=lambda *_: True)[0])
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert handlers.volumes == [40]
        asked = llm.requests[1][0][-2]
        assert json.loads(asked["tool_calls"][0]["function"]["arguments"]) == {"percent": 40}

    def test_the_tools_array_is_the_same_bytes_every_request(self) -> None:
        # It sits in the KV prefix right after the system prompt. A
        # reordered key is a full prefill every turn with no error.
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("battery")], ["<e:happy:7> Fine."], ["<e:happy:7> Ok."]
        )
        orch, _ = build_with_tools(llm, registry(handlers, confirm=lambda *_: True)[0])
        asyncio.run(orch.run(AUDIO))
        asyncio.run(orch.run(AUDIO))
        sent = [json.dumps(tools) for _, tools in llm.requests]
        assert len(sent) == 3 and sent[0]
        assert len(set(sent)) == 1
        assert [t["function"]["name"] for t in llm.requests[0][1]] == ["battery", "volume"]

    def test_a_registry_with_no_confirmer_offers_only_green_tools(self) -> None:
        # A tool the model can express but never complete is a turn that
        # ends in "not doing that" every time.
        handlers = Handlers()
        llm = ToolLlm(["<e:neutral:5> Hi."])
        orch, _ = build_with_tools(llm, registry(handlers, confirm=None)[0])
        asyncio.run(orch.run(AUDIO))
        assert [t["function"]["name"] for t in llm.requests[0][1]] == ["battery"]

    def test_no_registry_sends_no_tools(self) -> None:
        llm = ToolLlm(["<e:neutral:5> Hi."])
        orch, _ = build_with_tools(llm, None)
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert llm.requests[0][1] is None

    def test_a_denied_yellow_tool_does_not_run_and_the_turn_says_so(self, tmp_path) -> None:
        handlers = Handlers()
        llm = ToolLlm(["<e:neutral:5> ", call("volume", '{"percent": 30}')], ["<e:happy:7> Done."])
        reg, audit = registry(handlers, confirm=lambda *_: False, tmp_path=tmp_path)
        orch, parts = build_with_tools(llm, reg)
        result = asyncio.run(orch.run(AUDIO))

        assert handlers.volumes == []
        assert result.error == "ToolNotConfirmed" and not result.cancelled
        # No second request, nothing spoken from the model, nothing
        # remembered: the denial line is the caller's to speak.
        assert len(llm.requests) == 1
        assert parts["tts"].texts == []
        assert orch._history == []
        assert [e["event"] for e in audit.entries][-1] == DENIED
        assert audit.entries[0]["turn"] == result.turn.id

    def test_a_tool_that_raises_is_an_error_not_a_crash(self, tmp_path) -> None:
        handlers = Handlers()
        handlers.fail = OSError("no players found")
        llm = ToolLlm(["<e:neutral:5> ", call("battery")], ["<e:happy:7> Done."])
        reg, audit = registry(handlers, tmp_path=tmp_path)
        orch, parts = build_with_tools(llm, reg)
        result = asyncio.run(orch.run(AUDIO))

        assert result.error == "OSError" and not result.cancelled
        assert len(llm.requests) == 1
        assert parts["tts"].texts == []
        assert orch._history == []
        assert [e["event"] for e in audit.entries][-1] == FAILED

    def test_a_barge_in_during_a_tool_call_cancels_the_turn(self) -> None:
        # The call runs off the loop, so a barge-in lands while the tool
        # is still going: he does not wait for a six-second confirmation
        # window to end before she stops. A `registry.call()` made on the
        # loop instead blocks everything until the handler's timeout,
        # after which the turn finishes as if nothing happened.
        handlers = Handlers()
        handlers.block = threading.Event()
        llm = ToolLlm(["<e:neutral:5> ", call("battery")], ["<e:happy:7> Done."])
        orch, parts = build_with_tools(llm, registry(handlers)[0])

        async def go() -> tuple[TurnResult, bool]:
            turn = orch.begin_turn()
            task = asyncio.create_task(orch.run(AUDIO, turn=turn))
            while not handlers.started.is_set():
                await asyncio.sleep(0.005)
            turn.cancel.set()
            try:
                result = await asyncio.wait_for(task, timeout=1)
                return result, handlers.finished.is_set()
            finally:
                handlers.block.set()  # let the worker thread finish

        result, tool_had_finished = asyncio.run(go())
        assert result.cancelled
        assert not tool_had_finished, "the turn waited for the tool instead of stopping"
        assert len(llm.requests) == 1
        assert parts["sink"].cancelled == 1
        assert orch._history == []

    def test_a_rejected_call_is_fed_back_for_the_model_to_recover(self) -> None:
        # The registry's rejection names the real tools precisely so a
        # hallucinated name becomes a recoverable turn.
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("batery")],
            ["<e:neutral:5> ", call("battery")],
            ["<e:happy:7> Ninety six percent."],
        )
        orch, _ = build_with_tools(llm, registry(handlers)[0])
        result = asyncio.run(orch.run(AUDIO))

        assert result.error is None
        assert len(llm.requests) == 3
        rejection = llm.requests[1][0][-1]
        assert rejection["role"] == "tool" and "battery" in rejection["content"]
        assert handlers.battery_calls == 1
        assert result.reply == "Ninety six percent."

    def test_a_model_that_keeps_calling_tools_is_stopped(self) -> None:
        handlers = Handlers()
        llm = ToolLlm(*[["<e:neutral:5> ", call("battery")] for _ in range(MAX_TOOL_ROUNDS + 3)])
        orch, _ = build_with_tools(llm, registry(handlers)[0])
        result = asyncio.run(orch.run(AUDIO))
        assert result.error == "TooManyToolRounds"
        assert len(llm.requests) == MAX_TOOL_ROUNDS + 1
        assert orch._history == []

    def test_two_calls_in_one_round_run_in_order_with_their_own_ids(self) -> None:
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("battery"), call("volume", '{"percent": 20}')],
            ["<e:happy:7> Done both."],
        )
        orch, _ = build_with_tools(llm, registry(handlers, confirm=lambda *_: True)[0])
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert handlers.battery_calls == 1 and handlers.volumes == [20]
        asked, *answers = llm.requests[1][0][-3:]
        ids = [c["id"] for c in asked["tool_calls"]]
        assert len(set(ids)) == 2
        assert [a["tool_call_id"] for a in answers] == ids

    def test_history_trims_whole_exchanges(self) -> None:
        # A tool exchange is four messages. Cut in the middle it leaves a
        # `tool` message answering a call that is no longer there, which
        # the backends reject outright.
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5> ", call("battery")], ["<e:happy:7> Fine."], ["<e:happy:7> Ok."]
        )
        orch, _ = build_with_tools(llm, registry(handlers)[0])
        orch._history_limit = 1
        asyncio.run(orch.run(AUDIO))
        asyncio.run(orch.run(AUDIO))
        assert [m["role"] for m in orch._history] == ["user", "assistant"]
        assert orch._history[-1]["content"] == "Ok."

    def test_a_paced_stream_still_speaks_only_the_reply(self) -> None:
        # With real timing between events the voice keeps up with the
        # stream; a compliant tool round carries the tag and nothing
        # else, so nothing of it reaches the synthesiser either way.
        handlers = Handlers()
        llm = ToolLlm(
            ["<e:neutral:5>", call("battery")],
            ["<e:happy:7> Ninety ", "six percent. ", "Still charging."],
            pace=0.002,
        )
        orch, parts = build_with_tools(llm, registry(handlers)[0])
        result = asyncio.run(orch.run(AUDIO))
        assert result.error is None
        assert parts["tts"].texts == ["Ninety six percent.", "Still charging."]
        assert [s.label for s in parts["tts"].states] == [EmotionLabel.HAPPY] * 2
