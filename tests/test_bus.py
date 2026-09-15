"""Bounded queues, and the two things they must get right."""

from __future__ import annotations

import asyncio

import pytest

from elizabeth.bus import DEFAULT_DEPTH, Bus


class TestBounded:
    def test_an_unbounded_bus_cannot_be_created(self) -> None:
        # An unbounded queue is a memory leak with good manners.
        with pytest.raises(ValueError, match="leak"):
            Bus(depth=0)

    def test_the_default_lookahead_is_small(self) -> None:
        # Enough to keep the next stage busy; small enough that a
        # barge-in throws away almost nothing.
        assert 1 <= DEFAULT_DEPTH <= 4

    def test_put_blocks_when_the_consumer_is_behind(self) -> None:
        # The blocking IS the backpressure.
        async def go() -> bool:
            bus: Bus[int] = Bus(depth=1)
            await bus.put(1)
            try:
                await asyncio.wait_for(bus.put(2), timeout=0.05)
            except TimeoutError:
                return True
            return False

        assert asyncio.run(go())

    def test_nothing_is_dropped_under_pressure(self) -> None:
        # Dropping a sentence mid-reply is worse than pausing the
        # generator that produced it.
        async def go() -> list[int]:
            bus: Bus[int] = Bus(depth=2)
            got: list[int] = []

            async def produce() -> None:
                for i in range(10):
                    await bus.put(i)
                await bus.close()

            async def consume() -> None:
                async for item in bus:
                    await asyncio.sleep(0.001)
                    got.append(item)

            await asyncio.gather(produce(), consume())
            return got

        assert asyncio.run(go()) == list(range(10))


class TestClosing:
    def test_consumers_stop_when_the_producer_closes(self) -> None:
        async def go() -> list[int]:
            bus: Bus[int] = Bus()
            await bus.put(1)
            await bus.close()
            return [i async for i in bus]

        assert asyncio.run(go()) == [1]

    def test_close_is_idempotent_so_a_finally_can_call_it(self) -> None:
        async def go() -> bool:
            bus: Bus[int] = Bus()
            await bus.close()
            await bus.close()
            return bus.closed

        assert asyncio.run(go())

    def test_put_after_close_is_loud(self) -> None:
        async def go() -> None:
            bus: Bus[int] = Bus()
            await bus.close()
            await bus.put(1)

        with pytest.raises(RuntimeError, match="after close"):
            asyncio.run(go())

    def test_an_empty_stream_terminates(self) -> None:
        async def go() -> list[int]:
            bus: Bus[int] = Bus()
            await bus.close()
            return [i async for i in bus]

        assert asyncio.run(go()) == []


class TestBargeIn:
    def test_clear_drops_the_queued_reply(self) -> None:
        # Those sentences belong to a reply no longer being given.
        # Delivering them after the interruption is how an assistant
        # talks over the person who interrupted it.
        async def go() -> tuple[int, int]:
            bus: Bus[int] = Bus(depth=4)
            for i in range(3):
                await bus.put(i)
            return bus.clear(), bus.depth_now

        assert asyncio.run(go()) == (3, 0)

    def test_clearing_an_empty_bus_is_fine(self) -> None:
        async def go() -> int:
            return Bus().clear()

        assert asyncio.run(go()) == 0

    def test_depth_reports_how_far_behind_the_consumer_is(self) -> None:
        async def go() -> tuple[int, bool]:
            bus: Bus[int] = Bus(depth=2)
            await bus.put(1)
            await bus.put(2)
            return bus.depth_now, bus.full

        assert asyncio.run(go()) == (2, True)


class TestCloseNeverBlocks:
    def test_close_on_a_full_bus_returns_immediately(self) -> None:
        # The orchestrator's producer had exactly this shape — a `finally`
        # parking on `put(None)` into a full queue nobody would drain —
        # and it held the daemon's turn lock for the life of the process.
        async def go() -> list[int]:
            bus: Bus[int] = Bus(depth=1)
            await bus.put(1)
            await asyncio.wait_for(bus.close(), timeout=0.2)
            return [item async for item in bus]

        assert asyncio.run(go()) == [1]

    def test_a_consumer_parked_on_an_empty_bus_wakes_on_close(self) -> None:
        async def go() -> list[int]:
            bus: Bus[int] = Bus(depth=1)

            async def consume() -> list[int]:
                return [item async for item in bus]

            task = asyncio.create_task(consume())
            await asyncio.sleep(0)  # let it park on get()
            await bus.close()
            return await asyncio.wait_for(task, timeout=0.2)

        assert asyncio.run(go()) == []

    def test_clear_after_close_still_ends_the_stream(self) -> None:
        # Barge-in drops the queued reply; if that also dropped the
        # end-of-stream mark, the consumer would wait forever.
        async def go() -> list[int]:
            bus: Bus[int] = Bus(depth=2)
            await bus.put(1)
            await bus.close()

            async def consume() -> list[int]:
                return [item async for item in bus]

            task = asyncio.create_task(consume())
            await asyncio.sleep(0)
            bus.clear()
            return await asyncio.wait_for(task, timeout=0.2)

        assert asyncio.run(go()) in ([], [1])
