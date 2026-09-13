"""The WebSocket the browser talks to, and the protocol frozen in Stage 0.

**Why the browser owns audio playback.** Whoever owns the audio clock
owns the animation clock. If Python played the audio and the browser
animated the mouth, the two would drift and every viseme would need
re-syncing. Instead the browser holds one `AudioContext`, schedules
chunks against `ctx.currentTime`, and reports back the moment sequence 0
actually *played*. That timestamp is the only honest end of the one
metric this project measures — everything earlier is "we sent it".

**Why auth on localhost.** A localhost WebSocket is not private: any web
page the browser has open can `new WebSocket("ws://127.0.0.1:8760/…")`
and start listening to a microphone-driven assistant. So every
connection must present a per-process token AND an allowed `Origin`.
Both are checked before the socket is accepted, not after.

**Why the protocol is frozen now.** Later stages fill fields — visemes,
real emotions, user affect — but never change the shape. Writing it once,
with `visemes: []` and `neutral` placeholders, is what stops barge-in
being rewritten in Stage 2.
"""

# NOTE: no `from __future__ import annotations` here, deliberately.
# FastAPI resolves handler parameter types at decoration time; with
# PEP 563 the `WebSocket` annotation arrives as the STRING "WebSocket",
# which FastAPI cannot resolve, so it treats the parameter as a query
# parameter and rejects every handshake with 403 — with a valid token,
# from the right Origin, and no error anywhere. Verified: the endpoint
# was unreachable and `Session.check` was dead code.

import asyncio
import json
import secrets
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

# The session token rides the `Sec-WebSocket-Protocol` header as
# `neiro.token.<token>`, never the URL. uvicorn writes every handshake's
# path to the `uvicorn.error` logger at INFO — query string included,
# on accept AND on reject — so a token in the URL is a token in the
# journal, and `--no-access-log` does not help. Request headers are
# never logged, and this header is the one a browser lets a page set on
# a `WebSocket`. Verified against uvicorn 0.52.4's websockets impl.
TOKEN_SUBPROTOCOL_PREFIX = "neiro.token."

# Server -> client message types. Frozen; add fields, never rename.
SERVER_MESSAGES = (
    "hello",  # protocol version, session token echo, config the face needs
    "utt.begin",  # a reply is starting: emotion + how long to blend into it
    "utt.chunk",  # one audio chunk's metadata; the PCM follows as a binary frame
    "emotion",  # her state changed mid-reply (the tag resolves at ~token 6-9)
    "utt.end",
    "cancel",  # barge-in: stop now, drop what is queued
    "state",  # phase, input level, what we heard in his voice
)

# Client -> server.
CLIENT_MESSAGES = (
    "ready",  # the VRM loaded; here are the expressions it actually has
    "played",  # sequence N started playing at this AudioContext time
    "underrun",  # the queue ran dry mid-reply
    "error",
)


class ProtocolError(ValueError):
    """A message that does not match the frozen protocol."""


def audio_frame(audio_id: int, pcm: np.ndarray) -> bytes:
    """`uint32 audio_id || float32[] PCM @ 24 kHz`.

    Binary rather than JSON because base64 on every chunk is real CPU on
    the latency path, and the id lets the browser drop chunks belonging
    to a reply that was already cancelled — a late chunk from a
    cancelled turn arriving after the next turn started is otherwise a
    very confusing bug.
    """
    samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    return struct.pack("<I", int(audio_id)) + samples.tobytes()


def parse_audio_frame(payload: bytes) -> tuple[int, np.ndarray]:
    if len(payload) < 4:
        raise ProtocolError("audio frame shorter than its header")
    if (len(payload) - 4) % 4:
        raise ProtocolError("audio payload is not whole float32 samples")
    (audio_id,) = struct.unpack("<I", payload[:4])
    return audio_id, np.frombuffer(payload[4:], dtype=np.float32)


def server_message(kind: str, **fields: Any) -> str:
    if kind not in SERVER_MESSAGES:
        raise ProtocolError(f"unknown server message {kind!r}; frozen set is {SERVER_MESSAGES}")
    return json.dumps({"t": kind, **fields}, separators=(",", ":"))


def token_from_subprotocols(header: str | None) -> str | None:
    """The token a client offered as `neiro.token.<token>`, or None.

    The header is a comma-separated list; a browser may offer several
    and expects the server to select one of them. Anything without the
    prefix is ignored rather than rejected here — `Session.check` is
    the only place that says no, so it stays the only place to audit.
    """
    if not header:
        return None
    for offered in header.split(","):
        offered = offered.strip()
        if offered.startswith(TOKEN_SUBPROTOCOL_PREFIX):
            return offered[len(TOKEN_SUBPROTOCOL_PREFIX) :]
    return None


def _token_matches(presented: str, expected: str) -> bool:
    """Constant-time equality that cannot raise.

    `secrets.compare_digest` on two `str`s raises TypeError instead of
    returning False the moment either holds a non-ASCII character — and
    a header value decoded as latin-1 can hold any byte. Comparing as
    bytes keeps the constant-time property and turns "weird token" into
    "wrong token", which is what `Session.check` promises. `surrogatepass`
    is there so no `str` at all can make the encode itself raise.
    """
    return secrets.compare_digest(
        presented.encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass")
    )


def parse_client_message(raw: str) -> dict:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("not JSON") from exc
    if not isinstance(message, dict) or "t" not in message:
        raise ProtocolError("missing message type")
    if message["t"] not in CLIENT_MESSAGES:
        raise ProtocolError(f"unknown client message {message['t']!r}")
    return message


@dataclass
class Session:
    """One browser tab's connection.

    The token is minted per process, not per user: it exists to stop
    *other pages* opening the socket, and it dies with the daemon.
    """

    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    expressions: frozenset[str] = frozenset()
    ready: bool = False

    def check(self, token: str | None, origin: str | None, allowed_origins: set[str]) -> None:
        """Raise `PermissionError` unless this connection may proceed.

        Both checks, always. A token alone would let a malicious page
        that has somehow read the token connect; an Origin alone is
        forgeable by a non-browser client.
        """
        if not token or not _token_matches(token, self.token):
            raise PermissionError("bad or missing token")
        # A non-browser client sends no Origin at all. That is allowed
        # only because it cannot be a hostile *web page* — the token is
        # what protects that case.
        if origin is not None and origin not in allowed_origins:
            raise PermissionError(f"origin {origin!r} not allowed")


def build_app(session: Session, host: str = "127.0.0.1", port: int = 8760):
    """The FastAPI app. Imported lazily so the CLI stays fast to start."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    allowed_origins = {f"http://{host}:{port}", f"http://localhost:{port}"}
    app = FastAPI(title="neiro", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session
    app.state.outgoing: asyncio.Queue = asyncio.Queue(maxsize=64)

    @app.get("/health")
    async def health() -> dict:
        return {"protocol": PROTOCOL_VERSION, "ready": session.ready}

    @app.get("/")
    async def index() -> HTMLResponse:
        # The token is injected into the page rather than put in the URL:
        # URLs end up in history, in logs, and in the Referer header.
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>Neiro</title>"
            f"<script>window.NEIRO_TOKEN={json.dumps(session.token)};</script>"
            "<p>Neiro is running. The face lands in Stage 1.</p>"
        )

    @app.websocket("/neiro")
    async def endpoint(websocket: WebSocket) -> None:
        # Not the query string — see TOKEN_SUBPROTOCOL_PREFIX. `getlist`
        # because a client may split its offers across header lines.
        token = token_from_subprotocols(
            ", ".join(websocket.headers.getlist("sec-websocket-protocol"))
        )
        try:
            session.check(token, websocket.headers.get("origin"), allowed_origins)
        except PermissionError:
            # Close before accepting: an unauthenticated peer never gets
            # a usable socket, and learns nothing about why.
            await websocket.close(code=1008)
            return

        # A browser fails the handshake unless the server selects one of
        # the subprotocols it offered, so the token-bearing one is echoed
        # back — to the peer that already holds it, in a response header
        # nothing logs. `session.token`, not `token`: only a value that
        # passed the check is ever written into a response.
        await websocket.accept(subprotocol=TOKEN_SUBPROTOCOL_PREFIX + session.token)
        await websocket.send_text(
            server_message("hello", protocol=PROTOCOL_VERSION, samplerate=24000)
        )
        try:
            while True:
                message = parse_client_message(await websocket.receive_text())
                if message["t"] == "ready":
                    session.expressions = frozenset(message.get("expressions") or ())
                    session.ready = True
        except (WebSocketDisconnect, ProtocolError):
            session.ready = False

    return app
