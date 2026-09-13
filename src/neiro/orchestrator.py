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
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import numpy as np

from neiro.config import Neiro
from neiro.llm.chunker import SentenceChunker
from neiro.llm.emotion_tag import EmotionTagParser
from neiro.llm.prompt import system_message, user_message
from neiro.state import NEUTRAL_STATE, NeiroState, Turn, UserAffect

log = logging.getLogger(__name__)

# Bounded so a fast LLM cannot outrun a slower voice. Two chunks of
# lookahead is enough to keep the synthesiser busy and small enough that
# a barge-in throws away almost nothing.
CHUNK_QUEUE_DEPTH = 2


class Cancelled(Exception):
    """Raised internally when `turn.cancel` fires. Never escapes `run()`."""


@dataclass
class TurnResult:
    turn: Turn
    transcript: str = ""
    reply: str = ""
    state: NeiroState = field(default_factory=lambda: NEUTRAL_STATE)
    cancelled: bool = False
    error: str | None = None


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
    ) -> None:
        self.cfg = cfg or Neiro()
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sink = sink
        self.affect = affect
        self._history: list[dict] = []
        self._history_limit = history_limit
        self._turn_id = 0
        # The turn in flight, so barge-in can reach it. `None` between
        # turns; set by `begin_turn()` and cleared when `run()` returns.
        self.live: Turn | None = None

    # -- history --------------------------------------------------------

    def messages(self, transcript: str, annotation: str | None) -> list[dict]:
        """System prompt, then history, then this turn.

        The system message is byte-identical every turn and history is
        APPENDED, never rewritten. That is what makes the KV prefix cache
        hit; a single re-ordered message costs a full prefill (0.3-1.3 s)
        with no error to explain the latency.
        """
        return [system_message(), *self._history, user_message(transcript, annotation)]

    def _remember(self, transcript: str, annotation: str | None, reply: str) -> None:
        self._history.append(user_message(transcript, annotation))
        self._history.append({"role": "assistant", "content": reply})
        # Trim in PAIRS from the front. Dropping a lone user turn leaves
        # an assistant reply answering nothing, which reads as her having
        # hallucinated the question.
        while len(self._history) > self._history_limit * 2:
            del self._history[:2]

    def forget(self) -> None:
        self._history.clear()

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

    # -- the turn -------------------------------------------------------

    def begin_turn(self) -> Turn:
        """Create the Turn for the utterance now starting.

        Separate from `run()` because affect observes a rolling window
        *while he is still speaking* — before the endpoint, and therefore
        before `run()` is called. The caller needs the Turn during that
        window, so it cannot be created inside `run()`.
        """
        self._turn_id += 1
        self.live = Turn.new(turn_id=self._turn_id)
        return self.live

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
        self.live = turn
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

            parser = EmotionTagParser()
            chunker = SentenceChunker()
            reply_parts: list[str] = []
            state = NEUTRAL_STATE
            queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=CHUNK_QUEUE_DEPTH)

            async def produce() -> None:
                nonlocal state
                try:
                    async for event in self.llm.stream(self.messages(transcript, annotation)):
                        self._check(turn)
                        if "text" not in event:
                            continue
                        found, speakable = parser.feed(event["text"])
                        if found is not None:
                            state = found
                            turn.neiro_state = found
                            turn.stamp("emotion_resolved")
                        if not speakable:
                            continue
                        reply_parts.append(speakable)
                        for sentence in chunker.feed(speakable):
                            # Blocks when the voice is behind. That IS
                            # the backpressure.
                            await queue.put(sentence)
                    found, speakable = parser.flush()
                    if found is not None and turn.neiro_state is NEUTRAL_STATE:
                        state = found
                        turn.neiro_state = found
                    if speakable:
                        reply_parts.append(speakable)
                    for sentence in chunker.flush():
                        await queue.put(sentence)
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
            self._remember(transcript, annotation, result.reply)

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

        if self.affect is not None and not result.cancelled:
            # Once per utterance, never per window.
            self.affect.commit_utterance()
        self.live = None
        return result
