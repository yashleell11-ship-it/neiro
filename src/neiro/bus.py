"""Bounded queues between pipeline stages, and why they are bounded.

An unbounded queue is not a queue, it is a memory leak with good manners.
On this pipeline specifically, an unbounded chunk queue lets the LLM run
four hundred tokens ahead of the voice — and every one of those tokens
past an interruption is work thrown away, plus a reply she already
"decided" before hearing the correction.

So: `maxsize` everywhere, and a `put` that blocks. The blocking IS the
backpressure. There is no dropping policy here on purpose — dropping a
sentence mid-reply is worse than pausing the generator that produced it.

**Cancellation is first-class.** Every stage checks `turn.cancel` at its
awaits, so a queue also has to be drainable without waiting for a
producer that will never finish. `Bus.clear()` empties it and wakes
anyone blocked, which is what barge-in needs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

# Two items of lookahead: enough to keep the next stage busy while the
# previous one works, small enough that a barge-in throws away almost
# nothing. Measured intent, not a guess — see CHUNK_QUEUE_DEPTH in
# orchestrator.py, which this generalises.
DEFAULT_DEPTH = 2

# Sentinel for "the producer is finished". A None would collide with a
# legitimately empty payload.
_DONE = object()


@dataclass
class Bus[T]:
    """One bounded hop between two pipeline stages."""

    depth: int = DEFAULT_DEPTH
    name: str = "bus"
    _queue: asyncio.Queue = field(default=None, repr=False)
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"{self.name}: depth must be >= 1; an unbounded queue is a leak")
        self._queue = asyncio.Queue(maxsize=self.depth)

    async def put(self, item: T) -> None:
        """Blocks when the consumer is behind. That is the point."""
        if self._closed:
            raise RuntimeError(f"{self.name}: put after close")
        await self._queue.put(item)

    async def close(self) -> None:
        """Signal end-of-stream. Idempotent, so a `finally` can call it
        without checking — which is exactly where it belongs.

        Never blocks. The orchestrator's producer once had exactly the
        blocking shape (`await queue.put(None)` in a `finally`, into a
        full queue whose consumer had been cancelled) and it held the
        daemon's turn lock for the life of the process. If the queue is
        full the sentinel is simply not queued: `put` refuses new items
        after close, so the consumer can treat "closed and empty" as
        end-of-stream once it has drained what is there.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_DONE)
        except asyncio.QueueFull:
            pass

    def clear(self) -> int:
        """Drop everything queued. Returns how many were dropped.

        For barge-in: the queued sentences belong to a reply that is no
        longer being given, and delivering them after the interruption
        is how an assistant talks over the person who interrupted it.
        """
        dropped = 0
        while not self._queue.empty():
            try:
                if self._queue.get_nowait() is _DONE:
                    continue  # the sentinel is not part of the reply
                dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover
                break
        if self._closed:
            # A consumer parked on `get()` must still be woken; the
            # queue is empty now, so this cannot raise.
            self._queue.put_nowait(_DONE)
        return dropped

    async def __aiter__(self):
        """Consume until the producer closes."""
        while True:
            if self._closed and self._queue.empty():
                return
            item = await self._queue.get()
            if item is _DONE:
                return
            yield item

    @property
    def depth_now(self) -> int:
        """How far behind the consumer is — the HUD's "is she lagging"."""
        return self._queue.qsize()

    @property
    def full(self) -> bool:
        return self._queue.full()

    @property
    def closed(self) -> bool:
        return self._closed
