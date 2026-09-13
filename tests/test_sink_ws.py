"""The browser sink: the frozen protocol, frame by frame.

The orchestrator calls `play()` and `cancel()` and nothing else, so what
matters is what those two calls put on the wire — the exact sequence of
JSON and binary frames a two-chunk reply becomes, what a barge-in leaves
in the queue, and that the browser's `played` for sequence 0 lands on
the Turn as the end of the one metric.

The other thing worth a test is what the sink refuses to do: wait
forever. A tab that stops draining the queue must cost a dropped frame
and a log line, never a parked orchestrator — the daemon holds its lock
across the turn, and a parked orchestrator is the assistant dead until
restart (git log: 2296ba0).
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest

from neiro.audio.sink_ws import FIRST_AUDIO_TIMEOUT_S, SEND_TIMEOUT_S, WsSink
from neiro.config import Neiro
from neiro.metrics import headline_latency_ms
from neiro.server import (
    TOKEN_SUBPROTOCOL_PREFIX,
    Face,
    Session,
    _pump,
    parse_audio_frame,
)
from neiro.state import NEUTRAL_STATE, EmotionLabel, NeiroState, Turn

# Nothing here waits on a model; a test that hangs is a deadlock, and a
# deadlock must fail rather than stall the suite.
DEADLINE_S = 5.0


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, DEADLINE_S))


def _pcm(samples: int, level: float = 0.25) -> np.ndarray:
    return np.full(samples, level, dtype=np.float32)


def _entries(queue: asyncio.Queue) -> list:
    """The queue as the sink left it: a chunk is one (header, pcm) entry."""
    entries = []
    while True:
        try:
            entries.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return entries


def _drain(queue: asyncio.Queue) -> list:
    """The queue in wire order — what the pump would send, a chunk entry
    becoming its header then its PCM."""
    items = []
    for entry in _entries(queue):
        items.extend(entry if isinstance(entry, tuple) else (entry,))
    return items


def _kinds(items: list) -> list[str]:
    return [json.loads(item)["t"] if isinstance(item, str) else "audio" for item in items]


def _turn(label: EmotionLabel = EmotionLabel.HAPPY, intensity: float = 7 / 9) -> Turn:
    turn = Turn.new(3)
    turn.neiro_state = NeiroState.from_label(label, intensity)
    return turn


@pytest.fixture
def session() -> Session:
    """A session with a tab on it, as `Session.attach()` leaves it."""
    s = Session()
    s.attach()
    return s


class TestTwoChunkReply:
    def test_the_exact_frame_sequence(self, session: Session) -> None:
        cfg = Neiro()
        sink = WsSink(session, cfg)
        turn = _turn()

        async def go() -> None:
            await sink.play(
                turn,
                _pcm(2400),
                seq=0,
                text="Hi there.",
                visemes=[(0.0, "aa", 0.05), (0.05, "ih", 0.05)],
            )
            await sink.play(turn, _pcm(1200, 0.5), seq=1, text="Sit down.")
            sink.close()

        run(go())
        items = _drain(session.outgoing)
        assert _kinds(items) == ["utt.begin", "utt.chunk", "audio", "utt.chunk", "audio", "utt.end"]
        assert sink.dropped == 0

        begin = json.loads(items[0])
        audio_id = begin["audio_id"]
        assert audio_id == sink.audio_id
        assert begin["emotion"] == "happy"
        assert begin["intensity"] == pytest.approx(7 / 9)
        # The blend time is the blender's rise constant, from config —
        # the browser eases toward the target at the daemon's speed.
        assert begin["blend_ms"] == round(cfg.expression.tau_rise_s * 1000)
        assert begin["weights"]["happy"] == pytest.approx(7 / 9)
        assert all(begin["weights"][k] == 0 for k in ("angry", "sad", "relaxed", "surprised"))

        assert json.loads(items[1]) == {
            "t": "utt.chunk",
            "seq": 0,
            "audio_id": audio_id,
            "dur_ms": 100,
            "text": "Hi there.",
            "visemes": [[0.0, "aa", 0.05], [0.05, "ih", 0.05]],
        }
        frame_id, pcm = parse_audio_frame(items[2])
        assert frame_id == audio_id
        assert np.array_equal(pcm, _pcm(2400))

        assert json.loads(items[3]) == {
            "t": "utt.chunk",
            "seq": 1,
            "audio_id": audio_id,
            "dur_ms": 50,
            "text": "Sit down.",
            "visemes": [],
        }
        frame_id, pcm = parse_audio_frame(items[4])
        assert frame_id == audio_id
        assert np.array_equal(pcm, _pcm(1200, 0.5))

        assert json.loads(items[5]) == {"t": "utt.end", "audio_id": audio_id}

    def test_each_reply_gets_its_own_audio_id(self, session: Session) -> None:
        # A late chunk from the previous reply must be droppable by id.
        first, second = WsSink(session), WsSink(session)

        async def go() -> None:
            await first.play(_turn(), _pcm(240))
            await second.play(_turn(), _pcm(240))

        run(go())
        assert first.audio_id != second.audio_id
        ids = {json.loads(i)["audio_id"] for i in _drain(session.outgoing) if isinstance(i, str)}
        assert ids == {first.audio_id, second.audio_id}

    def test_close_without_a_chunk_sends_nothing(self, session: Session) -> None:
        # A turn that produced no audio (empty transcript, LLM down)
        # has no utterance to end.
        WsSink(session).close()
        assert session.outgoing.empty()


class TestEmotionMidReply:
    def test_a_changed_state_between_chunks_is_an_emotion_frame(self, session: Session) -> None:
        sink = WsSink(session)
        turn = Turn.new(1)  # NEUTRAL until the tag resolves

        async def go() -> None:
            await sink.play(turn, _pcm(240), seq=0)
            turn.neiro_state = NeiroState.from_label(EmotionLabel.SAD, 0.5)
            await sink.play(turn, _pcm(240), seq=1)
            await sink.play(turn, _pcm(240), seq=2)  # unchanged: no frame

        run(go())
        items = _drain(session.outgoing)
        assert _kinds(items) == [
            "utt.begin",
            "utt.chunk",
            "audio",
            "emotion",
            "utt.chunk",
            "audio",
            "utt.chunk",
            "audio",
        ]
        begin, emotion = json.loads(items[0]), json.loads(items[3])
        assert begin["emotion"] == NEUTRAL_STATE.label.value
        assert all(w == 0 for w in begin["weights"].values())
        assert emotion["emotion"] == "sad"
        assert emotion["weights"]["sad"] == pytest.approx(0.5)
        assert emotion["audio_id"] == begin["audio_id"]

    def test_weights_target_what_the_avatar_can_actually_do(self, session: Session) -> None:
        # The page said in `ready` that it has no `surprised`; the
        # blender's fallback sends the surprise to `happy` rather than
        # to a preset that would silently do nothing.
        session.expressions = frozenset({"happy", "sad", "neutral"})
        sink = WsSink(session)
        run(sink.play(_turn(EmotionLabel.SURPRISED, 0.9), _pcm(240)))
        begin = json.loads(_drain(session.outgoing)[0])
        assert begin["emotion"] == "surprised"
        assert begin["weights"]["surprised"] == 0
        assert begin["weights"]["happy"] == pytest.approx(0.9)


class TestCancelMidReply:
    def test_everything_queued_is_dropped_and_cancel_is_sent(self, session: Session) -> None:
        sink = WsSink(session)
        turn = _turn()

        async def go() -> None:
            await sink.play(turn, _pcm(2400), seq=0, text="First.")
            await sink.play(turn, _pcm(2400), seq=1, text="Second.")
            # The tab has drained nothing yet — the worst case for a
            # barge-in, since all of it would still be played.
            await sink.cancel(turn)
            sink.close()  # after a cancel there is nothing to end

        run(go())
        items = _drain(session.outgoing)
        assert _kinds(items) == ["cancel"]
        assert json.loads(items[0]) == {"t": "cancel", "audio_id": sink.audio_id}
        assert sink.cancelled

    def test_cancel_before_any_chunk_sends_nothing(self, session: Session) -> None:
        sink = WsSink(session)
        run(sink.cancel(_turn()))
        assert session.outgoing.empty()
        assert sink.cancelled

    def test_cancel_gets_through_a_full_queue(self) -> None:
        # The one frame that must never be dropped, and a full queue is
        # exactly when it is sent: dropping the queued reply first is
        # what makes room.
        session = Session(outgoing=asyncio.Queue(maxsize=1))
        session.attach()
        sink = WsSink(session)
        turn = _turn()

        async def go() -> None:
            await sink.play(turn, _pcm(240), seq=0)  # begin fills the one slot
            await sink.cancel(turn)

        run(go())
        assert _kinds(_drain(session.outgoing)) == ["cancel"]

    def test_terminate_is_the_talk_loops_barge_in_after_the_turn(self, session: Session) -> None:
        # SPACE during the tail of a finished reply: the loop kills
        # "playback", which for this sink means the browser's queue.
        sink = WsSink(session)
        run(sink.play(_turn(), _pcm(240)))
        sink.close()
        _drain(session.outgoing)
        sink.terminate()
        assert _kinds(_drain(session.outgoing)) == ["cancel"]
        assert sink.poll() == 0


class TestNothingWaitsForever:
    def test_no_tab_means_dropped_not_parked(self) -> None:
        session = Session()  # never attached
        sink = WsSink(session)

        async def go() -> float:
            started = time.perf_counter()
            await sink.play(_turn(), _pcm(2400), seq=0, text="Hello?")
            return time.perf_counter() - started

        elapsed = run(go())
        assert elapsed < SEND_TIMEOUT_S
        assert session.outgoing.empty()
        assert sink.dropped >= 1 and sink.sent == 0

    def test_a_tab_that_stops_draining_costs_a_frame_not_the_pipeline(self) -> None:
        session = Session(outgoing=asyncio.Queue(maxsize=1))
        session.attach()
        sink = WsSink(session)

        async def go() -> float:
            started = time.perf_counter()
            await sink.play(_turn(), _pcm(2400), seq=0, text="Hello?")
            return time.perf_counter() - started

        elapsed = run(go())
        # Waited the bound for room, then gave up — did not wait longer.
        assert SEND_TIMEOUT_S * 0.9 <= elapsed < SEND_TIMEOUT_S * 3
        assert sink.dropped >= 1
        # The chunk — header and PCM, one entry — did not fit; the entry
        # that did is the one before it.
        assert _kinds(_drain(session.outgoing)) == ["utt.begin"]

    def test_a_reconnecting_tab_does_not_get_the_old_replys_tail(self, session: Session) -> None:
        sink = WsSink(session)
        run(sink.play(_turn(), _pcm(240)))
        assert not session.outgoing.empty()
        session.detach()
        session.attach()  # a fresh tab never saw utt.begin; the rest is garbage
        assert session.outgoing.empty()


class TestHeaderAndPcmAreOneEntry:
    """The browser pairs each binary frame with the header before it, so
    a header queued without its PCM would pair with the NEXT chunk's
    audio and every later chunk of the reply would play with the wrong
    seq, text and visemes. The sink must therefore never leave a header
    on the queue whose PCM was dropped — and it cannot, because the two
    are one entry.
    """

    def test_a_chunk_is_one_entry_so_two_free_slots_are_not_needed(self) -> None:
        # Room for exactly two entries: utt.begin and one chunk. Queued
        # as separate frames the header would fit and the PCM would not,
        # leaving a header with nothing behind it.
        session = Session(outgoing=asyncio.Queue(maxsize=2))
        session.attach()
        sink = WsSink(session)
        turn = _turn()
        run(sink.play(turn, _pcm(2400), seq=0, text="Hi."))

        assert sink.dropped == 0
        entries = _entries(session.outgoing)
        assert len(entries) == 2
        header, pcm = entries[1]
        assert json.loads(header)["seq"] == 0
        assert parse_audio_frame(pcm)[0] == json.loads(header)["audio_id"]

    def test_a_full_queue_drops_the_whole_chunk_never_a_bare_header(self) -> None:
        session = Session(outgoing=asyncio.Queue(maxsize=2))
        session.attach()
        sink = WsSink(session)
        turn = _turn()

        async def go() -> float:
            await sink.play(turn, _pcm(2400), seq=0, text="First.")  # fills both slots
            started = time.perf_counter()
            await sink.play(turn, _pcm(2400), seq=1, text="Second.")  # no room: dropped
            return time.perf_counter() - started

        elapsed = run(go())
        assert SEND_TIMEOUT_S * 0.9 <= elapsed < SEND_TIMEOUT_S * 3
        assert sink.dropped == 1
        entries = _entries(session.outgoing)
        # Not one bare header anywhere: a text entry is never utt.chunk.
        assert all(isinstance(e, tuple) or json.loads(e)["t"] != "utt.chunk" for e in entries)
        wire = _kinds([i for e in entries for i in (e if isinstance(e, tuple) else (e,))])
        assert wire == ["utt.begin", "utt.chunk", "audio"]

    def test_the_pump_sends_header_then_pcm_with_nothing_between(self) -> None:
        # A barge-in that lands while the header is on its way out must
        # still follow the PCM on the wire, not slip between the two.
        session = Session()
        session.attach()
        sink = WsSink(session)
        turn = _turn()
        sent: list = []

        class Socket:
            async def send_text(self, text: str) -> None:
                sent.append(text)
                if json.loads(text)["t"] == "utt.chunk":
                    await sink.cancel(turn)  # queued now, sent after the PCM
                await asyncio.sleep(0)

            async def send_bytes(self, data: bytes) -> None:
                sent.append(data)
                await asyncio.sleep(0)

        async def go() -> None:
            await sink.play(turn, _pcm(2400), seq=0, text="Hi.")
            pump = asyncio.create_task(_pump(session, Socket()))
            while len(sent) < 4:
                await asyncio.sleep(0.005)
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

        run(go())
        assert _kinds(sent) == ["utt.begin", "utt.chunk", "audio", "cancel"]


class TestPlayedEndsTheMetric:
    def test_played_for_seq_0_stamps_the_turn_once(self, session: Session) -> None:
        sink = WsSink(session)
        turn = _turn()
        turn.stamp("endpoint")

        async def go() -> None:
            await sink.play(turn, _pcm(240), seq=0)
            audio_id = sink.audio_id
            assert headline_latency_ms(turn) is None  # sent is not played
            # A later chunk, and another reply's seq 0: neither is ours.
            session.receive({"t": "played", "seq": 1, "at": 0.2, "audio_id": audio_id})
            session.receive({"t": "played", "seq": 0, "at": 0.2, "audio_id": audio_id + 1})
            assert not await sink.wait_played(0.05)
            assert headline_latency_ms(turn) is None

            session.receive({"t": "played", "seq": 0, "at": 0.25, "audio_id": audio_id})
            assert await sink.wait_played(1.0)
            first = turn.timeline["sink_played"]
            assert headline_latency_ms(turn) > 0
            # A repeat must not move the metric's end.
            session.receive({"t": "played", "seq": 0, "at": 0.9, "audio_id": audio_id})
            assert turn.timeline["sink_played"] == first

        run(go())

    def test_wait_played_is_false_for_a_reply_that_never_started(self, session: Session) -> None:
        sink = WsSink(session)
        assert not run(sink.wait_played(0.05))

    def test_poll_says_speaking_until_the_sent_audio_has_played(self, session: Session) -> None:
        sink = WsSink(session)
        turn = _turn()

        async def go() -> None:
            await sink.play(turn, _pcm(240), seq=0)  # 10 ms of audio
            sink.close()
            assert sink.poll() is None  # sent, not yet reported playing
            session.receive({"t": "played", "seq": 0, "at": 0.0, "audio_id": sink.audio_id})
            await asyncio.sleep(0.05)  # longer than what was sent
            assert sink.poll() == 0

        run(go())

    def test_poll_gives_up_on_a_tab_that_never_reports_played(
        self, session: Session, monkeypatch
    ) -> None:
        # The tab has the frames but its AudioContext is suspended and
        # the unlock button is never clicked: no `played`, ever. The
        # sink says "speaking" for as long as the HUD would wait for the
        # report, then stops — the loop must not carry a reply nobody
        # heard as "speaking" until the next keypress.
        now = [1000.0]
        monkeypatch.setattr(time, "perf_counter", lambda: now[0])
        sink = WsSink(session)
        turn = _turn()

        async def go() -> None:
            await sink.play(turn, _pcm(2400), seq=0)
            assert sink.poll() is None  # still being sent: no close yet
            sink.close()
            assert sink.poll() is None
            now[0] += FIRST_AUDIO_TIMEOUT_S * 0.9
            assert sink.poll() is None  # within the allowance
            now[0] += FIRST_AUDIO_TIMEOUT_S * 0.2
            assert sink.poll() == 0  # given up on
            assert headline_latency_ms(turn) is None  # and no number was invented

        run(go())

    def test_poll_is_done_when_the_tab_goes_away(self, session: Session) -> None:
        sink = WsSink(session)
        run(sink.play(_turn(), _pcm(2400), seq=0))
        sink.close()
        assert sink.poll() is None
        session.detach()
        assert sink.poll() == 0


class TestEndToEnd:
    """The whole path, on one event loop, the way `neiro talk --browser`
    runs it: uvicorn as a task beside the caller, a real WebSocket
    client where the tab would be, the sink fed a fake turn, and the
    browser-side frames read back in order — then `played` sent from
    the client side and found on the Turn.

    To rerun by hand:
        PYTHONPATH=src .venv/bin/python -m pytest tests/test_sink_ws.py -k EndToEnd -p no:cacheprovider
    """

    def test_a_fake_turn_reaches_the_client_in_order_and_played_comes_back(self, tmp_path) -> None:
        import websockets

        cfg = Neiro()
        # Port 0: whatever is free. tmp_path as web/: no page and no
        # avatar, so `hello` must say so rather than guess.
        face = Face(cfg, port=0, web_dir=tmp_path)

        async def go() -> None:
            server = asyncio.create_task(face.serve())
            try:
                await face.wait_started(DEADLINE_S)
                async with websockets.connect(
                    f"ws://127.0.0.1:{face.port}/neiro",
                    subprotocols=[TOKEN_SUBPROTOCOL_PREFIX + face.session.token],
                ) as ws:
                    hello = json.loads(await ws.recv())
                    assert hello["t"] == "hello"
                    assert hello["samplerate"] == cfg.audio.output_samplerate
                    assert hello["face"] == cfg.expression.model_dump()
                    assert hello["avatar"] is None

                    await ws.send(
                        json.dumps({"t": "ready", "expressions": ["happy", "sad", "neutral"]})
                    )
                    while not face.session.ready:
                        await asyncio.sleep(0.01)

                    sink = WsSink(face.session, cfg)
                    turn = _turn(EmotionLabel.SURPRISED, 0.9)
                    turn.stamp("endpoint")
                    await sink.play(turn, _pcm(2400), seq=0, text="Oh.", visemes=[(0.0, "oh", 0.1)])
                    await sink.play(turn, _pcm(1200), seq=1, text="Hi.")
                    sink.close()

                    frames = [await ws.recv() for _ in range(6)]
                    assert _kinds(frames) == [
                        "utt.begin",
                        "utt.chunk",
                        "audio",
                        "utt.chunk",
                        "audio",
                        "utt.end",
                    ]
                    begin = json.loads(frames[0])
                    # `ready` said there is no `surprised`: redistributed.
                    assert begin["weights"]["happy"] == pytest.approx(0.9)
                    audio_id, pcm = parse_audio_frame(frames[2])
                    assert audio_id == begin["audio_id"]
                    assert np.array_equal(pcm, _pcm(2400))
                    assert json.loads(frames[1])["visemes"] == [[0.0, "oh", 0.1]]

                    assert headline_latency_ms(turn) is None
                    await ws.send(
                        json.dumps({"t": "played", "seq": 0, "at": 0.51, "audio_id": audio_id})
                    )
                    assert await sink.wait_played(DEADLINE_S)
                    assert headline_latency_ms(turn) > 0
                    assert sink.dropped == 0
            finally:
                server.cancel()
                await asyncio.gather(server, return_exceptions=True)

        asyncio.run(asyncio.wait_for(go(), DEADLINE_S * 3))
        # Cancelling the task was a real shutdown: nothing listens now.
        import socket

        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", face.port), timeout=1.0).close()

    def test_a_taken_port_fails_at_bind_not_inside_the_server_task(self, tmp_path) -> None:
        first = Face(Neiro(), port=0, web_dir=tmp_path)
        first.bind()
        try:
            second = Face(Neiro(), port=first.port, web_dir=tmp_path)
            with pytest.raises(OSError):
                second.bind()
        finally:
            first.close()
