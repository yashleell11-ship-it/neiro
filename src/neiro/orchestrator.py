"""The turn: audio in, her voice out.

Four stages handing each other a `Turn` by reference, connected by
bounded queues. Bounded is the point — without backpressure the LLM runs
four hundred tokens ahead of the voice, and everything it generated past
the interruption has to be thrown away anyway.

**`turn.cancel` is checked at every await, from the first line.** Not
added in Stage 2 when barge-in lands. Retrofitting cancellation into a
pipeline is how you get a half-cancelled turn: audio stopped, LLM still
streaming, history containing words she never said. The event exists
from day one and every stage respects it, so Stage 2 only has to decide
*when* to set it.

**Affect runs during speech, not after.** `observe()` is called on a
rolling window while he is still talking, so the prosody reading costs
the turn budget nothing. By the time the endpoint fires the answer is
already there.

**The timeline is the cheapest useful thing here.** Every stage stamps
`turn.timeline`, and that dict is what `metrics.py` turns into the one
measured number: endpoint → the browser's `played` for sequence 0.

**Tools are a second request, not a second code path.** When a stream
ends in tool calls the registry runs them and the same producer asks
the model again with the results appended — the request is the previous
one plus two messages, so the KV prefix still hits. What the model said
*alongside* a call is never her reply: the tag from that stream drives
her face while the tool runs, the prose is narration the prompt forbids,
and the tag on the stream that follows the result is the one that
drives her voice. The clients only surface calls when a stream is done,
so suppression means what it can: nothing queued but unspoken survives,
and nothing from a tool round is remembered as something she said.

**Narration that reached the voice is a design limit, not a bug to
fix here.** Real suppression would mean holding every sentence of
every turn until its stream ends, which is the whole first-audio
budget spent on the rare turn that needs it; and a one-token hold
would not do — live, the 4B's "Let me check the battery, memory, and
disk space for you." left the chunker as two comma-split pieces, the
first released while the second was still streaming. The prompt is
the lever (v5, "Doing things"), and how often the model still narrates
or skips a call it should make is a Stage 3 compliance measurement,
alongside tag compliance.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import numpy as np

from neiro.config import Neiro
from neiro.llm.chunker import SentenceChunker
from neiro.llm.emotion_tag import EmotionTagParser
from neiro.llm.prompt import system_message, user_message
from neiro.state import NEUTRAL_STATE, NeiroState, ToolCall, Turn, UserAffect
from neiro.tools.registry import ToolRegistry, ToolRejected
from neiro.tools.tiers import Tier as ToolTier

log = logging.getLogger(__name__)

# Bounded so a fast LLM cannot outrun a slower voice. Two chunks of
# lookahead is enough to keep the synthesiser busy and small enough that
# a barge-in throws away almost nothing.
CHUNK_QUEUE_DEPTH = 2

# How many consecutive requests may end in tool calls before the turn
# gives up. Two is the longest chain a builtin tool documents —
# `list_windows` then `focus_window` — and a model still asking for
# tools after that is looping, not working. Structural, like the queue
# depth: a model that needs more rounds needs a different tool, not a
# bigger number.
MAX_TOOL_ROUNDS = 2


class Cancelled(Exception):
    """Raised internally when `turn.cancel` fires. Never escapes `run()`."""


class TooManyToolRounds(RuntimeError):
    """The model asked for tools on more consecutive requests than
    `MAX_TOOL_ROUNDS` allows. Spoken as a tool failure.
    """


@dataclass
class TurnResult:
    turn: Turn
    transcript: str = ""
    reply: str = ""
    state: NeiroState = field(default_factory=lambda: NEUTRAL_STATE)
    cancelled: bool = False
    error: str | None = None


def tool_call_message(turn_id: int, round_no: int, calls: list[ToolCall]) -> dict:
    """The assistant message that made these calls, in the OpenAI shape.

    `content` is empty on purpose. Anything the model said alongside a
    call was narration the prompt forbids and the voice did not carry,
    and history is what he heard — the same rule that clips an
    interrupted reply. Ids are deterministic so the bytes of a
    remembered exchange never depend on when it was remembered.
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{turn_id}_{round_no}_{call.index}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.args, separators=(",", ":")),
                },
            }
            for call in calls
        ],
    }


def tool_result_message(call_id: str, content: str) -> dict:
    """One tool's result, in the shape the assistant message's id binds it to."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class Orchestrator:
    """One turn at a time. Providers are injected, never imported here,
    so the whole loop can be exercised with fakes and no hardware.
    """

    def __init__(
        self,
        stt,
        llm,
        tts,
        sink,
        affect=None,
        cfg: Neiro | None = None,
        history_limit: int = 12,
        *,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.cfg = cfg or Neiro()
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sink = sink
        self.affect = affect
        self.tools = tools
        # Built once and sent as the same object every turn. The tools
        # array sits in the KV prefix right after the system prompt;
        # regenerating it per request would be correct today and one
        # refactor away from a reordered key, which is a full prefill
        # every turn with no error to explain it. With no way to
        # confirm, YELLOW tools are not offered at all.
        self._tool_schemas: list[dict] = (
            tools.schemas(None if tools.can_confirm else ToolTier.GREEN) if tools else []
        )
        self._history: list[dict] = []
        # One entry per exchange, holding how many messages it added, so
        # trimming can drop a whole exchange: a tool exchange is four or
        # more messages and cutting it in the middle leaves a `tool`
        # message answering a call that is no longer there — which the
        # backends reject outright.
        self._exchange_sizes: list[int] = []
        self._history_limit = history_limit
        self._turn_id = 0

    # -- history --------------------------------------------------------

    def messages(self, transcript: str, annotation: str | None) -> list[dict]:
        """System prompt, then history, then this turn.

        The system message is byte-identical every turn and history is
        APPENDED, never rewritten. That is what makes the KV prefix cache
        hit; a single re-ordered message costs a full prefill (0.3-1.3 s)
        with no error to explain the latency.
        """
        return [system_message(), *self._history, user_message(transcript, annotation)]

    def tool_schemas(self) -> list[dict]:
        """What the model is offered, identical every turn."""
        return self._tool_schemas

    def _remember(self, exchange: list[dict]) -> None:
        """Append one finished exchange — the user turn, any tool calls
        and their results, and her reply — and trim whole exchanges from
        the front. Dropping a lone user turn leaves an assistant reply
        answering nothing, which reads as her having hallucinated the
        question.
        """
        self._history.extend(exchange)
        self._exchange_sizes.append(len(exchange))
        while len(self._exchange_sizes) > self._history_limit:
            del self._history[: self._exchange_sizes.pop(0)]

    def forget(self) -> None:
        self._history.clear()
        self._exchange_sizes.clear()

    # -- stages ---------------------------------------------------------

    @staticmethod
    def _check(turn: Turn) -> None:
        if turn.cancel.is_set():
            raise Cancelled

    @staticmethod
    async def _next_or_cancelled(turn: Turn, queue: asyncio.Queue[str | None]) -> str | None:
        """`queue.get()` that a barge-in can wake.

        Between the STT result and the LLM's first token this is the
        only await in the turn, and a plain `get()` cannot notice
        `turn.cancel`: a check placed after it only runs once an item
        arrives, and the first check that could fire otherwise lives
        inside `produce()`'s stream loop, which needs an LLM event to
        run at all. So a barge-in during a cold prefill waited for the
        model he had just interrupted to start talking — measured with
        a 2 s time-to-first-token fake, cancel at 0.1 s was acted on at
        2.0 s — and the bound is the provider's HTTP timeout, 120 s for
        Ollama. `Daemon.handle()` holds its lock across `run()`, so his
        new utterance could not start for that whole interval.

        Racing the get against the event makes the wait itself the
        cancel point. Both tasks are cancelled on every exit; a get
        cancelled after a `put()` landed leaves the item on the queue,
        where `run()`'s drain throws it away with the rest of the reply
        that is no longer being given.
        """
        getter = asyncio.ensure_future(queue.get())
        interrupted = asyncio.ensure_future(turn.cancel.wait())
        try:
            done, _ = await asyncio.wait({getter, interrupted}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            interrupted.cancel()
            getter.cancel()  # a no-op once it holds the item
        if interrupted in done:
            raise Cancelled
        return getter.result()

    async def observe_while_speaking(self, turn: Turn, window: np.ndarray) -> UserAffect:
        """Called repeatedly WHILE he is still talking, so this costs the
        turn budget nothing. Never after the endpoint.
        """
        if self.affect is None:
            return UserAffect.NONE
        turn.user_affect = await self.affect.observe(window)
        return turn.user_affect

    async def transcribe(self, turn: Turn) -> str:
        self._check(turn)
        turn.stamp("stt_start")
        transcript = await self.stt.transcribe(turn.audio)
        turn.transcript = transcript
        turn.stamp("stt_done")
        # After the await as well as before it: a barge-in that landed
        # while STT was running means the LLM is never asked. Without
        # this the request went out and was cancelled a token later.
        self._check(turn)
        return transcript

    async def speak(
        self, turn: Turn, chunks: AsyncIterator[str], state_ref: Callable[[], NeiroState]
    ) -> None:
        """Synthesise and play, sentence by sentence.

        Sequence 0 is stamped separately because it is the end of the one
        metric — everything after it is already audible.
        """
        seq = 0
        async for text in chunks:
            self._check(turn)
            async for pcm, visemes in self.tts.synth(text, state_ref()):
                self._check(turn)
                if seq == 0:
                    turn.stamp("tts_first_chunk")
                await self.sink.play(turn, pcm, seq=seq, text=text, visemes=visemes)
                if seq == 0:
                    turn.stamp("sink_first_sent")
                seq += 1

    async def run_tool(self, turn: Turn, call: ToolCall) -> str:
        """One call through the registry, off the event loop.

        `registry.call()` blocks: a YELLOW tool waits up to the
        confirmation window for him to click, and every handler talks
        to a subprocess or a socket. On the loop that would freeze
        affect windows and barge-in for the duration — and a barge-in
        would not even be noticed until the tool returned. So it runs in
        a worker thread, and the await is where a cancel lands: the
        voice side notices `turn.cancel` and `run()` cancels the
        producer, which is parked right here. The thread finishes on its
        own and its result is never spoken; the audit log still records
        what it did.

        A rejected call comes back as the tool's result rather than as a
        failure: the registry's message names the real tools precisely
        so the model can recover on the next round.
        """
        if self.tools is None:
            raise RuntimeError(f"the model called {call.name!r} but no tool registry is wired")
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(None, self.tools.call, call.name, call.args, turn.id)
        # An abandoned worker that later raises would log "exception was
        # never retrieved" at garbage-collection time; retrieving it is
        # the whole handler.
        worker.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            return await worker
        except ToolRejected as exc:
            return str(exc)

    # -- the turn -------------------------------------------------------

    def begin_turn(self) -> Turn:
        """Create the Turn for the utterance now starting.

        Separate from `run()` because affect observes a rolling window
        *while he is still speaking* — before the endpoint, and therefore
        before `run()` is called. The caller needs the Turn during that
        window, so it cannot be created inside `run()`.
        """
        self._turn_id += 1
        return Turn.new(turn_id=self._turn_id)

    async def run(
        self, audio: np.ndarray, annotation: str | None = None, turn: Turn | None = None
    ) -> TurnResult:
        """One complete turn. Never raises for an expected failure —
        the caller gets a `TurnResult` with `error` set and Neiro says
        something in character about it.

        `turn` is the one `begin_turn()` returned, if the caller was
        observing affect during speech. Omitted, a fresh one is made —
        which is the CLI's case, where there is no speaking phase to
        observe.
        """
        turn = turn if turn is not None else self.begin_turn()
        turn.audio = audio
        turn.stamp("endpoint")
        result = TurnResult(turn=turn)

        try:
            transcript = await self.transcribe(turn)
            if not transcript.strip():
                # The STT confidence floor already rejected it. Saying
                # nothing would look like she froze.
                result.error = "empty_transcript"
                return result
            result.transcript = transcript

            reply_parts: list[str] = []
            state = NEUTRAL_STATE
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=CHUNK_QUEUE_DEPTH)
            messages = self.messages(transcript, annotation)
            # What this turn will append to history if it finishes: the
            # user turn, then whatever the rounds below add.
            exchange: list[dict] = [messages[-1]]

            def feel(found: NeiroState) -> None:
                nonlocal state
                state = found
                turn.neiro_state = found
                if "emotion_resolved" not in turn.timeline:
                    # The first tag is the one the waterfall measures:
                    # it is when her face could start moving.
                    turn.stamp("emotion_resolved")

            async def one_round(tools: list[dict] | None) -> tuple[list[str], list[ToolCall]]:
                """Stream one request. Returns the speakable parts it
                produced and the tool calls it ended in; sentences go to
                the voice as they complete, except the chunker's tail,
                which the caller releases only once it knows this was
                not a tool round.
                """
                parser = EmotionTagParser()
                chunker = SentenceChunker()
                parts: list[str] = []
                accumulator = None
                # A copy: the list grows between rounds, and a client
                # that held the reference would see the next round's
                # messages appear inside the request it already sent.
                async for event in self.llm.stream(list(messages), tools=tools):
                    self._check(turn)
                    if "done" in event:
                        accumulator = event["done"]
                    if "text" not in event:
                        continue
                    found, speakable = parser.feed(event["text"])
                    if found is not None:
                        feel(found)
                    if not speakable:
                        continue
                    parts.append(speakable)
                    for sentence in chunker.feed(speakable):
                        # Blocks when the voice is behind. That IS
                        # the backpressure.
                        await queue.put(sentence)
                found, speakable = parser.flush()
                if found is not None and turn.neiro_state is NEUTRAL_STATE:
                    feel(found)
                if speakable:
                    parts.append(speakable)
                calls = accumulator.tool_calls() if accumulator is not None else []
                if not calls:
                    for sentence in chunker.flush():
                        await queue.put(sentence)
                return parts, calls

            async def produce() -> None:
                try:
                    tools = self._tool_schemas or None
                    for round_no in range(MAX_TOOL_ROUNDS + 1):
                        parts, calls = await one_round(tools)
                        if not calls:
                            reply_parts.extend(parts)
                            exchange.append(
                                {"role": "assistant", "content": "".join(parts).strip()}
                            )
                            return
                        if round_no == MAX_TOOL_ROUNDS:
                            raise TooManyToolRounds(
                                f"still calling tools after {MAX_TOOL_ROUNDS} rounds"
                            )
                        # A tool round. Whatever prose it produced was
                        # narration: drop what the voice has not taken
                        # yet, and never count it as her reply.
                        while not queue.empty():
                            queue.get_nowait()
                        turn.tool_calls.extend(calls)
                        asked = tool_call_message(turn.id, round_no, calls)
                        messages.append(asked)
                        exchange.append(asked)
                        for call, requested in zip(calls, asked["tool_calls"], strict=True):
                            self._check(turn)
                            answer = tool_result_message(
                                requested["id"], await self.run_tool(turn, call)
                            )
                            messages.append(answer)
                            exchange.append(answer)
                finally:
                    await queue.put(None)

            async def consume() -> AsyncIterator[str]:
                while True:
                    # Between sentences is a legitimate cancel point —
                    # and so is the wait for the next one, which is where
                    # the consumer sits for the whole of the LLM's
                    # time-to-first-token. See `_next_or_cancelled`.
                    item = await self._next_or_cancelled(turn, queue)
                    if item is None:
                        return
                    yield item

            producer = asyncio.create_task(produce())
            try:
                await self.speak(turn, consume(), lambda: state)
            finally:
                # Always await the producer, even on cancel: a detached
                # task would keep streaming into a queue nobody reads and
                # hold the HTTP connection open.
                producer.cancel()
                # ...but DRAIN FIRST, or that await never returns.
                #
                # `produce()`'s own `finally` does `await queue.put(None)`.
                # When cancellation reaches the producer while it is
                # parked on a FULL queue — the designed steady state,
                # since the queue being full IS the backpressure — that
                # cleanup starts a *fresh* put on a queue nobody will
                # drain again. asyncio does not re-deliver cancellation
                # to a task that suspends inside its own finally, so the
                # producer parks forever and this gather never returns.
                #
                # In production `Daemon.handle()` holds its lock across
                # this call, so the whole assistant is dead until
                # restart — on the barge-in path this module exists for.
                # Reproduced before fixing; see tests/test_orchestrator.py.
                #
                # The queued sentences are garbage at this point by
                # design: they belong to a reply that is no longer being
                # given.
                while not queue.empty():
                    queue.get_nowait()
                await asyncio.gather(producer, return_exceptions=True)

            # The producer's failure IS the turn's failure. Its `finally`
            # puts the None sentinel on the queue, so `speak()` finishes
            # perfectly happily on an empty stream — which meant a dead
            # llama-server produced a successful turn with an empty reply
            # and no error anywhere. Exactly the silent failure this
            # project exists to design out; found by a test, not in use.
            if not producer.cancelled() and producer.exception() is not None:
                raise producer.exception()

            result.reply = "".join(reply_parts).strip()
            result.state = state
            turn.stamp("turn_done")
            self._remember(exchange)

        except Cancelled:
            result.cancelled = True
            # The partial reply is CLIPPED from history. Remembering
            # words she was interrupted before saying makes her refer
            # back to things he never heard.
            await self.sink.cancel(turn)
            turn.stamp("cancelled")
        except Exception as exc:
            log.exception("turn %d failed", turn.id)
            result.error = type(exc).__name__
            turn.stamp("failed")
        finally:
            # A `finally`, because the empty-transcript path returns from
            # inside the `try` and used to skip both of these. The affect
            # provider had already staged that utterance's last window and
            # hysteresis history during observe(); with no commit to close
            # them they leaked into the next turn — its band was decided
            # against the rejected utterance's history, and its features
            # were folded into his baseline one turn late. And `live`
            # stayed pointing at a finished turn, so the next barge-in
            # "cancelled" it and reported success.
            if self.affect is not None:
                if result.cancelled or result.error == "empty_transcript":
                    # Not a sample of how he normally sounds: nothing
                    # usable was said, or it was cut off. Dropped, so it
                    # neither enters the baseline nor decides the next
                    # turn's band. A downstream failure (LLM down) still
                    # commits — he spoke normally; she just could not answer.
                    self.affect.discard_utterance()
                else:
                    # Once per utterance, never per window.
                    self.affect.commit_utterance()
        return result
