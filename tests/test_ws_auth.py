"""The WebSocket protocol and its auth.

A localhost WebSocket is NOT private: any page the browser has open can
`new WebSocket("ws://127.0.0.1:8760/neiro")` and start listening to a
microphone-driven assistant. These tests are the wall.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
import pytest

from neiro.server import (
    CLIENT_MESSAGES,
    PROTOCOL_VERSION,
    SERVER_MESSAGES,
    TOKEN_SUBPROTOCOL_PREFIX,
    ProtocolError,
    Session,
    audio_frame,
    build_app,
    parse_audio_frame,
    parse_client_message,
    server_message,
    token_from_subprotocols,
)

HOST, PORT = "127.0.0.1", 8760
GOOD_ORIGIN = f"http://{HOST}:{PORT}"
ORIGINS = {GOOD_ORIGIN, f"http://localhost:{PORT}"}

# The app runs on another thread under both clients below, so anything
# it writes to the session is observed with a bounded wait. A deadline,
# not a sleep: the assertion still fails if the write never happens.
SETTLE_DEADLINE_S = 2.0
SETTLE_POLL_S = 0.005


def _wait_until(predicate: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + SETTLE_DEADLINE_S
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"{what} did not happen within {SETTLE_DEADLINE_S}s")
        time.sleep(SETTLE_POLL_S)


class TestAuth:
    def test_the_right_token_and_origin_pass(self) -> None:
        s = Session()
        s.check(s.token, "http://127.0.0.1:8760", ORIGINS)

    def test_no_token_is_refused(self) -> None:
        s = Session()
        with pytest.raises(PermissionError):
            s.check(None, "http://127.0.0.1:8760", ORIGINS)
        with pytest.raises(PermissionError):
            s.check("", "http://127.0.0.1:8760", ORIGINS)

    def test_a_wrong_token_is_refused(self) -> None:
        s = Session()
        with pytest.raises(PermissionError):
            s.check("guessed", "http://127.0.0.1:8760", ORIGINS)

    @pytest.mark.parametrize("weird", ["\u00e9", "\U0001f642", "\udcff"])
    def test_a_weird_token_is_refused_not_a_crash(self, weird: str) -> None:
        # `secrets.compare_digest` on two strs raises TypeError, not
        # False, for any non-ASCII character — and a header decoded as
        # latin-1 can hold any byte. The contract is PermissionError.
        # The lone surrogate is the case a plain `.encode()` would turn
        # into a UnicodeEncodeError instead.
        s = Session()
        with pytest.raises(PermissionError):
            s.check(weird, GOOD_ORIGIN, ORIGINS)
        with pytest.raises(PermissionError):
            s.check(s.token + weird, GOOD_ORIGIN, ORIGINS)

    def test_a_hostile_page_with_the_token_is_still_refused(self) -> None:
        # Defence in depth: both checks, always. A token could leak.
        s = Session()
        with pytest.raises(PermissionError, match="origin"):
            s.check(s.token, "https://evil.example", ORIGINS)

    def test_a_non_browser_client_sends_no_origin_and_is_allowed(self) -> None:
        # It cannot be a hostile web PAGE; the token protects that case.
        s = Session()
        s.check(s.token, None, ORIGINS)

    def test_tokens_differ_between_sessions(self) -> None:
        assert Session().token != Session().token

    def test_the_token_is_long_enough_not_to_be_guessed(self) -> None:
        assert len(Session().token) >= 24


class TestProtocolIsFrozen:
    def test_the_stage_0_message_set_is_complete(self) -> None:
        # Later stages fill fields; they never rename or drop a type.
        # That is what stops barge-in being rewritten in Stage 2.
        assert set(SERVER_MESSAGES) == {
            "hello",
            "utt.begin",
            "utt.chunk",
            "emotion",
            "utt.end",
            "cancel",
            "state",
        }
        assert set(CLIENT_MESSAGES) == {"ready", "played", "underrun", "error"}

    def test_an_unknown_server_message_cannot_be_sent(self) -> None:
        with pytest.raises(ProtocolError):
            server_message("utt.oops")

    def test_a_known_message_serialises_compactly(self) -> None:
        raw = server_message("utt.begin", emotion="happy", blend_ms=120)
        assert json.loads(raw) == {"t": "utt.begin", "emotion": "happy", "blend_ms": 120}
        assert " " not in raw  # no wasted bytes on the latency path

    @pytest.mark.parametrize("bad", ['{"t":"nope"}', "not json", "[]", '{"no_type":1}'])
    def test_junk_from_the_client_is_rejected(self, bad: str) -> None:
        with pytest.raises(ProtocolError):
            parse_client_message(bad)

    def test_a_valid_client_message_parses(self) -> None:
        msg = parse_client_message('{"t":"played","seq":0,"at":1.25}')
        assert msg["t"] == "played" and msg["seq"] == 0

    def test_protocol_version_is_declared(self) -> None:
        assert PROTOCOL_VERSION >= 1


class TestAudioFrames:
    def test_round_trip(self) -> None:
        pcm = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        audio_id, back = parse_audio_frame(audio_frame(7, pcm))
        assert audio_id == 7
        assert np.allclose(back, pcm)

    def test_the_id_lets_a_cancelled_reply_be_dropped(self) -> None:
        # A late chunk from a cancelled turn arriving after the next turn
        # started is otherwise a very confusing bug.
        first, _ = parse_audio_frame(audio_frame(1, np.zeros(4, dtype=np.float32)))
        second, _ = parse_audio_frame(audio_frame(2, np.zeros(4, dtype=np.float32)))
        assert first != second

    def test_an_empty_chunk_is_legal(self) -> None:
        audio_id, pcm = parse_audio_frame(audio_frame(3, np.zeros(0, dtype=np.float32)))
        assert audio_id == 3 and pcm.size == 0

    def test_a_truncated_frame_is_rejected(self) -> None:
        with pytest.raises(ProtocolError):
            parse_audio_frame(b"\x01\x02")

    def test_a_misaligned_payload_is_rejected(self) -> None:
        # Not whole float32 samples — silently reinterpreting this would
        # produce noise that sounds like a codec bug.
        with pytest.raises(ProtocolError):
            parse_audio_frame(audio_frame(1, np.zeros(4, dtype=np.float32)) + b"\x00")

    def test_it_is_binary_not_base64(self) -> None:
        # base64 on every chunk is real CPU on the latency path.
        pcm = np.zeros(240, dtype=np.float32)
        assert len(audio_frame(0, pcm)) == 4 + 240 * 4


class TestTokenSubprotocol:
    """`Sec-WebSocket-Protocol` is the one request header a browser lets
    a page set on a WebSocket, so the token rides there as
    `neiro.token.<token>` — never in the URL, which uvicorn logs."""

    def test_no_header_is_no_token(self) -> None:
        assert token_from_subprotocols(None) is None
        assert token_from_subprotocols("") is None

    def test_the_prefixed_offer_is_found_among_others(self) -> None:
        # A browser may offer several; the list is comma-separated with
        # optional whitespace, and ours need not be first.
        assert token_from_subprotocols(f"chat, {TOKEN_SUBPROTOCOL_PREFIX}abc ,other") == "abc"

    def test_an_unrelated_offer_is_not_a_token(self) -> None:
        assert token_from_subprotocols("chat") is None
        assert token_from_subprotocols("neiro.tokenabc") is None

    def test_a_bare_prefix_is_an_empty_token_which_check_refuses(self) -> None:
        empty = token_from_subprotocols(TOKEN_SUBPROTOCOL_PREFIX)
        assert empty == ""
        with pytest.raises(PermissionError):
            Session().check(empty, GOOD_ORIGIN, ORIGINS)


class TestTheEndpoint:
    """`build_app`'s wiring, driven through a real ASGI client.

    `Session.check` is a predicate and TestAuth covers it. These tests
    are about what feeds it and what happens around it: where the token
    and the Origin are read from, that a refusal closes BEFORE
    `accept()`, that the allowed origins come from host and port, and
    that `ready` lands in the session. Swapping the two arguments at the
    call site, widening the origin set, reading the token from the query
    string again, or closing after accept would each fail nothing above.
    """

    def _client(self, session: Session):
        from starlette.testclient import TestClient

        return TestClient(build_app(session, host=HOST, port=PORT))

    def _connect(
        self, client, *, token: str | bytes | None, origin: str | None, path: str = "/neiro"
    ):
        headers = {} if origin is None else {"Origin": origin}
        offered = None
        if isinstance(token, bytes):
            # Sent verbatim. httpx refuses a non-ASCII *str* header value
            # before it leaves the client, but the wire carries bytes and
            # Starlette decodes them as latin-1 — so this is the only way
            # to put a non-ASCII token in front of the server.
            headers["sec-websocket-protocol"] = TOKEN_SUBPROTOCOL_PREFIX.encode() + token
        elif token is not None:
            offered = [TOKEN_SUBPROTOCOL_PREFIX + token]
        return client.websocket_connect(path, subprotocols=offered, headers=headers)

    def _refused_before_accept(self, client, **how) -> None:
        from starlette.websockets import WebSocketDisconnect

        # The connect's `__enter__` raises only if the first frame is a
        # close; had the app accepted first, the body would run and fail.
        with pytest.raises(WebSocketDisconnect) as refusal, self._connect(client, **how):
            pytest.fail("the socket was accepted")
        assert refusal.value.code == 1008

    def test_the_right_token_and_origin_get_hello(self) -> None:
        session = Session()
        with self._connect(self._client(session), token=session.token, origin=GOOD_ORIGIN) as ws:
            hello = json.loads(ws.receive_text())
        assert hello["t"] == "hello"
        assert hello["protocol"] == PROTOCOL_VERSION

    def test_the_server_selects_the_subprotocol_the_page_offered(self) -> None:
        # A browser aborts the handshake unless the server picks one of
        # the subprotocols the page offered; a bare accept() is a
        # connection that opens on the wire and dies in the browser.
        session = Session()
        with self._connect(self._client(session), token=session.token, origin=GOOD_ORIGIN) as ws:
            assert ws.accepted_subprotocol == TOKEN_SUBPROTOCOL_PREFIX + session.token

    def test_localhost_is_the_other_allowed_spelling(self) -> None:
        session = Session()
        client = self._client(session)
        with self._connect(client, token=session.token, origin=f"http://localhost:{PORT}") as ws:
            assert json.loads(ws.receive_text())["t"] == "hello"

    def test_a_non_browser_client_sends_no_origin_and_gets_hello(self) -> None:
        session = Session()
        with self._connect(self._client(session), token=session.token, origin=None) as ws:
            assert json.loads(ws.receive_text())["t"] == "hello"

    def test_no_token_is_closed_before_accept(self) -> None:
        self._refused_before_accept(self._client(Session()), token=None, origin=GOOD_ORIGIN)

    def test_a_wrong_token_is_closed_before_accept(self) -> None:
        self._refused_before_accept(self._client(Session()), token="guessed", origin=GOOD_ORIGIN)

    def test_a_non_ascii_token_is_closed_before_accept_not_a_traceback(self) -> None:
        # The endpoint catches PermissionError and nothing else; a
        # TypeError out of the comparison would be an unhandled ASGI
        # exception per attempt — log flooding on demand.
        self._refused_before_accept(self._client(Session()), token=b"\xe9", origin=GOOD_ORIGIN)

    def test_the_right_token_from_a_hostile_origin_is_closed_before_accept(self) -> None:
        session = Session()
        self._refused_before_accept(
            self._client(session), token=session.token, origin="https://evil.example"
        )

    def test_the_same_host_on_another_port_is_another_origin(self) -> None:
        session = Session()
        self._refused_before_accept(
            self._client(session), token=session.token, origin=f"http://{HOST}:{PORT + 1}"
        )

    def test_a_token_in_the_query_string_is_not_honoured(self) -> None:
        # The old transport. Honouring it would put the token back in
        # uvicorn's log line on every handshake.
        session = Session()
        self._refused_before_accept(
            self._client(session),
            token=None,
            origin=GOOD_ORIGIN,
            path=f"/neiro?token={session.token}",
        )

    def test_ready_carries_the_avatars_real_expressions(self) -> None:
        # `surprised` is often unbound in real VRM models; the blender
        # needs to know before it targets one.
        session = Session()
        assert not session.ready
        with self._connect(self._client(session), token=session.token, origin=GOOD_ORIGIN) as ws:
            ws.receive_text()  # hello
            ws.send_text(json.dumps({"t": "ready", "expressions": ["happy", "sad", "neutral"]}))
            _wait_until(lambda: session.ready, "ready")
            assert session.expressions == frozenset({"happy", "sad", "neutral"})
            assert "surprised" not in session.expressions

    def test_ready_without_expressions_is_an_empty_set_not_a_crash(self) -> None:
        session = Session()
        with self._connect(self._client(session), token=session.token, origin=GOOD_ORIGIN) as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"t": "ready"}))
            _wait_until(lambda: session.ready, "ready")
            assert session.expressions == frozenset()

    def test_disconnecting_clears_ready(self) -> None:
        session = Session()
        with self._connect(self._client(session), token=session.token, origin=GOOD_ORIGIN) as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"t": "ready", "expressions": ["happy"]}))
            _wait_until(lambda: session.ready, "ready")
            ws.close()
            _wait_until(lambda: not session.ready, "ready clearing on disconnect")


class TestUnderUvicorn:
    """One real handshake through uvicorn, because uvicorn is where the
    token used to leak: it logs every handshake's path — query string
    included — on `uvicorn.error` at INFO, accepted or refused. Nothing
    that drives the ASGI app directly can see that line.

    It is also the layer that caught the PEP 563 bug: `from __future__
    import annotations` made FastAPI see the `websocket: WebSocket`
    parameter as the *string* "WebSocket", treat it as a query
    parameter, and answer every handshake with 403 — valid token, right
    Origin, no error anywhere, `Session.check` dead code — while every
    unit test here passed.
    """

    def _serve(self):
        import threading

        import uvicorn

        session = Session()
        # log_config=None: the default config calls dictConfig on the
        # process-wide logging tree, which every later test would inherit.
        server = uvicorn.Server(
            uvicorn.Config(
                build_app(session, host=HOST, port=PORT), host=HOST, port=0, log_config=None
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait_until(lambda: server.started, "uvicorn startup")
        port = server.servers[0].sockets[0].getsockname()[1]
        return session, server, thread, port

    def test_a_real_handshake_gets_hello_and_never_logs_the_token(self) -> None:
        import asyncio
        import logging

        import websockets

        captured: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = captured.append  # type: ignore[method-assign]
        log = logging.getLogger("uvicorn.error")
        previous_level = log.level
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        session, server, thread, port = self._serve()

        async def go() -> str:
            async with websockets.connect(
                f"ws://{HOST}:{port}/neiro",
                subprotocols=[TOKEN_SUBPROTOCOL_PREFIX + session.token],
                additional_headers={"Origin": GOOD_ORIGIN},
            ) as ws:
                return await asyncio.wait_for(ws.recv(), SETTLE_DEADLINE_S)

        try:
            hello = json.loads(asyncio.run(go()))
            assert hello["t"] == "hello"
            assert hello["protocol"] == PROTOCOL_VERSION
        finally:
            server.should_exit = True
            thread.join(SETTLE_DEADLINE_S)
            log.removeHandler(handler)
            log.setLevel(previous_level)

        lines = [record.getMessage() for record in captured]
        handshakes = [line for line in lines if "WebSocket /neiro" in line]
        # The line must exist, or the assertion after it could never fail.
        assert handshakes and all("[accepted]" in line for line in handshakes), lines
        assert session.token not in "\n".join(lines)

    def test_the_module_does_not_use_pep_563(self) -> None:
        # The specific thing that broke it. A future refactor adding the
        # import back would silently kill the endpoint again.
        from pathlib import Path as P

        # Checked as a STATEMENT, not a substring: the comment in
        # server.py explaining why it is absent naturally contains the
        # words, and matching those would make this pass for the wrong
        # reason -- or fail when someone documents the decision.
        source = (P(__file__).resolve().parents[1] / "src/neiro/server.py").read_text()
        statements = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        assert not any(line.strip() == "from __future__ import annotations" for line in statements)
