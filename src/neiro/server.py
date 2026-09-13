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

**Why the server runs inside the talk loop's event loop.** The sink
(`audio/sink_ws.py`) puts frames on a queue and the endpoint's pump
takes them off; both sides on one loop means an `asyncio.Queue` is the
whole hand-off — no threads, no locks, and the browser's `played`
message lands on the very `Turn` object the orchestrator is still
holding. `Face` binds the port itself before uvicorn starts, because
uvicorn answers a busy port with `sys.exit()` from inside its task, and
a `SystemExit` raised inside an asyncio task takes the whole loop down
with a traceback that names nothing useful.
"""

# NOTE: no `from __future__ import annotations` here, deliberately.
# FastAPI resolves handler parameter types at decoration time; with
# PEP 563 the `WebSocket` annotation arrives as the STRING "WebSocket",
# which FastAPI cannot resolve, so it treats the parameter as a query
# parameter and rejects every handshake with 403 — with a valid token,
# from the right Origin, and no error anywhere. Verified: the endpoint
# was unreachable and `Session.check` was dead code.

import asyncio
import contextlib
import json
import logging
import secrets
import socket
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neiro.config import Neiro
from neiro.state import Turn

log = logging.getLogger(__name__)

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

# Where `neiro talk --browser` listens. Loopback only, always: the token
# and Origin checks are the wall against other *pages*, not against
# other machines, and nothing here is built to face a network.
HOST = "127.0.0.1"
DEFAULT_PORT = 8760

# The page, and everything it imports, served from the repo's web/ tree.
# The avatar directory is gitignored (see ASSETS.md); a VRM dropped there
# is picked up on the next connection, no config needed.
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
AVATAR_DIR = "public/avatar"

# The token is written into the page rather than put in the URL, and the
# page carries this marker where it goes. Served with `no-store` so the
# token never sits in the browser's disk cache either.
TOKEN_MARKER = "<!-- neiro:token -->"

# Frames buffered for one tab. Two per chunk (the JSON header and the
# PCM), so this is ~32 chunks — several sentences of lookahead. A tab
# further behind than that is not rendering, and the sink drops rather
# than waits (see audio/sink_ws.py).
OUTGOING_DEPTH = 64

# `played` for sequence 0 arrives *after* the sink has finished with a
# short reply — a one-chunk answer is entirely sent before it is heard —
# so a few recent replies stay addressable by audio_id.
REMEMBERED_REPLIES = 4

# How long a graceful shutdown may wait for the tab to answer the close
# frame before the connection is dropped. A closed laptop lid must not
# hold `neiro talk` open on exit.
SHUTDOWN_TIMEOUT_S = 1.0

# Close codes. 1008 = policy violation (auth), 1003 = unsupported data
# (the peer spoke something that is not the protocol), 1013 = try again
# later (a second tab while the first still owns the audio clock).
CLOSE_POLICY = 1008
CLOSE_UNSUPPORTED = 1003
CLOSE_BUSY = 1013


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


def face_config(cfg: Neiro) -> dict[str, float]:
    """What the page needs to blend expressions the way `emotion/blend.py`
    does: the asymmetric time constants and the clamps, from config, so
    the browser never carries its own copy of a number that is tuned
    here.
    """
    return cfg.expression.model_dump()


def avatar_url(web_dir: Path) -> str | None:
    """The first VRM under `web/public/avatar/`, as a path the static
    mount serves, or None — in which case the page draws its fallback
    face and says so. Nothing is invented: no file, no avatar.
    """
    folder = web_dir / AVATAR_DIR
    if not folder.is_dir():
        return None
    for path in sorted(folder.rglob("*.vrm")):
        return path.relative_to(web_dir).as_posix()
    return None


@dataclass
class _Reply:
    """One reply the browser may still report `played` for."""

    turn: Turn
    played: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Session:
    """One browser tab's connection.

    The token is minted per process, not per user: it exists to stop
    *other pages* opening the socket, and it dies with the daemon.

    It is also the hand-off between the sink and the socket: the sink
    puts frames on `outgoing`, the endpoint's pump sends them, and the
    browser's `played` comes back through `receive()` onto the Turn the
    sink registered with `begin_reply()`.
    """

    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    expressions: frozenset[str] = frozenset()
    ready: bool = False
    connected: bool = False
    outgoing: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTGOING_DEPTH), repr=False
    )
    _replies: dict[int, _Reply] = field(default_factory=dict, repr=False)
    _next_audio_id: int = field(default=0, repr=False)

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

    # -- the tab ----------------------------------------------------------

    def attach(self) -> None:
        """A tab connected. Frames queued for the tab that is gone belong
        to a reply this one never saw begin; they are dropped, not
        replayed.
        """
        self.discard_queued()
        self.connected = True

    def detach(self) -> None:
        self.connected = False
        self.ready = False
        self.expressions = frozenset()

    def push(self, item: str | bytes) -> bool:
        """Queue one frame without waiting. False when the tab is behind
        or absent — the caller decides whether that is worth a log line.
        """
        if not self.connected:
            return False
        try:
            self.outgoing.put_nowait(item)
        except asyncio.QueueFull:
            return False
        return True

    def discard_queued(self) -> int:
        """Barge-in: everything not yet sent is garbage. Returns how much."""
        dropped = 0
        while True:
            try:
                self.outgoing.get_nowait()
            except asyncio.QueueEmpty:
                return dropped
            dropped += 1

    # -- replies ----------------------------------------------------------

    def begin_reply(self, turn: Turn) -> int:
        """Mint the audio_id for a reply and remember its Turn, so the
        browser's `played` for sequence 0 can stamp the right one.
        """
        self._next_audio_id += 1
        self._replies[self._next_audio_id] = _Reply(turn=turn)
        while len(self._replies) > REMEMBERED_REPLIES:
            del self._replies[next(iter(self._replies))]
        return self._next_audio_id

    async def wait_played(self, audio_id: int, timeout: float) -> bool:
        """True once the browser reported sequence 0 of this reply
        playing; False if it has not within `timeout` (no tab, a muted
        context, a reply that was cancelled first).
        """
        reply = self._replies.get(audio_id)
        if reply is None:
            return False
        try:
            await asyncio.wait_for(reply.played.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def receive(self, message: dict) -> None:
        """One parsed client message. `played` for sequence 0 is the end
        of the one metric and lands on the Turn itself — stamped here,
        at receipt, because this is the first moment Python can know.
        """
        kind = message["t"]
        if kind == "ready":
            self.expressions = frozenset(message.get("expressions") or ())
            self.ready = True
        elif kind == "played":
            reply = self._replies.get(message.get("audio_id"))
            if reply is None or message.get("seq") != 0 or reply.played.is_set():
                # A reply we no longer track, a later chunk, or a repeat:
                # none of them may move the metric's end.
                return
            reply.turn.stamp("sink_played")
            reply.played.set()
        elif kind == "underrun":
            log.warning("browser audio queue ran dry at seq %s", message.get("seq"))
        elif kind == "error":
            log.error("browser reported: %s", message.get("message"))


async def _pump(session: Session, websocket) -> None:
    """Outgoing frames to the socket, in queue order. Text is protocol
    JSON, bytes are `audio_frame()`s; the order they were queued in is
    the order the browser needs them (a chunk's header precedes its PCM).
    """
    while True:
        item = await session.outgoing.get()
        if isinstance(item, bytes):
            await websocket.send_bytes(item)
        else:
            await websocket.send_text(item)


def build_app(
    session: Session,
    host: str = HOST,
    port: int = DEFAULT_PORT,
    cfg: Neiro | None = None,
    web_dir: Path = WEB_DIR,
):
    """The FastAPI app. Imported lazily so the CLI stays fast to start."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    cfg = cfg or Neiro()
    allowed_origins = {f"http://{host}:{port}", f"http://localhost:{port}"}
    app = FastAPI(title="neiro", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session
    index_path = web_dir / "index.html"

    @app.get("/health")
    async def health() -> dict:
        return {"protocol": PROTOCOL_VERSION, "ready": session.ready}

    @app.get("/")
    async def index() -> HTMLResponse:
        # The token is injected into the page rather than put in the URL:
        # URLs end up in history, in logs, and in the Referer header.
        inject = f"<script>window.NEIRO_TOKEN={json.dumps(session.token)};</script>"
        if index_path.is_file():
            page = index_path.read_text(encoding="utf-8")
            if TOKEN_MARKER in page:
                page = page.replace(TOKEN_MARKER, inject, 1)
            else:
                page = inject + page
        else:
            page = (
                "<!doctype html><meta charset=utf-8><title>Neiro</title>"
                f"{inject}<p>Neiro is running, but web/index.html is missing.</p>"
            )
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

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
            await websocket.close(code=CLOSE_POLICY)
            return
        if session.connected:
            # One tab owns the audio clock. Two would each get half the
            # frames, and both would sound broken in a way no log
            # explains. A refresh is fine: the old socket closes first.
            await websocket.close(code=CLOSE_BUSY)
            return

        # A browser fails the handshake unless the server selects one of
        # the subprotocols it offered, so the token-bearing one is echoed
        # back — to the peer that already holds it, in a response header
        # nothing logs. `session.token`, not `token`: only a value that
        # passed the check is ever written into a response.
        await websocket.accept(subprotocol=TOKEN_SUBPROTOCOL_PREFIX + session.token)
        await websocket.send_text(
            server_message(
                "hello",
                protocol=PROTOCOL_VERSION,
                samplerate=cfg.audio.output_samplerate,
                face=face_config(cfg),
                avatar=avatar_url(web_dir),
            )
        )
        session.attach()
        pump = asyncio.create_task(_pump(session, websocket))
        try:
            while True:
                session.receive(parse_client_message(await websocket.receive_text()))
        except WebSocketDisconnect:
            pass
        except ProtocolError as exc:
            # Something that is not the page is on the other end. Say so
            # on the wire and stop, rather than leave a socket open that
            # nobody reads from.
            log.warning("closing browser socket: %s", exc)
            with contextlib.suppress(Exception):
                await websocket.close(code=CLOSE_UNSUPPORTED)
        finally:
            session.detach()
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

    if web_dir.is_dir():
        # After the explicit routes, so `/` stays the token-injecting
        # page and the mount only answers what nothing above did.
        app.mount("/", StaticFiles(directory=web_dir), name="web")

    return app


class Face:
    """The served face: one session, the app, and uvicorn in-process.

    `bind()` claims the port synchronously so `neiro talk --browser` can
    print a URL that works and fail loudly when the port is taken —
    before the loop, the mic and the raw terminal exist. `serve()` is
    then a coroutine the talk loop runs beside itself and cancels on
    exit; cancellation becomes uvicorn's own graceful shutdown, bounded
    by `SHUTDOWN_TIMEOUT_S`.
    """

    def __init__(
        self,
        cfg: Neiro | None = None,
        host: str = HOST,
        port: int = DEFAULT_PORT,
        web_dir: Path = WEB_DIR,
    ) -> None:
        self.cfg = cfg or Neiro()
        self.host = host
        self.port = port
        self.web_dir = web_dir
        self.session = Session()
        self.app = None
        self._sock: socket.socket | None = None
        self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def bind(self) -> str:
        """Claim the port now. Raises `OSError` if it is taken — as a
        normal exception the CLI can print, not a `SystemExit` from
        inside a task. Port 0 asks the kernel for a free one, and the
        app's allowed origins are built from the port actually bound.
        """
        import uvicorn

        if self._sock is not None:
            return self.url
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
            # Listening now, not when uvicorn starts: with SO_REUSEADDR a
            # second bind to a port that is bound but not yet listening
            # succeeds on Linux, and the collision would surface later,
            # inside the server task. uvicorn calls listen() again with
            # its own backlog; that is fine.
            sock.listen()
        except OSError:
            sock.close()
            raise
        self.port = sock.getsockname()[1]
        self._sock = sock
        self.app = build_app(self.session, self.host, self.port, self.cfg, self.web_dir)
        # log_config=None: uvicorn's default calls dictConfig on the
        # process-wide logging tree, and the talk loop's console would
        # inherit it. The graceful-shutdown bound keeps a tab that never
        # answers the close frame from holding exit open.
        self._server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_config=None,
                timeout_graceful_shutdown=SHUTDOWN_TIMEOUT_S,
            )
        )
        return self.url

    def close(self) -> None:
        """Release the port without ever having served — the CLI failing
        between `bind()` and the loop, or a test that only wanted to know
        the port was claimed.
        """
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def started(self) -> bool:
        return self._server is not None and bool(self._server.started)

    async def wait_started(self, timeout: float) -> None:
        """Until uvicorn is accepting connections; TimeoutError otherwise."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.started:
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"the face server did not start within {timeout}s")
            await asyncio.sleep(0.01)

    async def serve(self) -> None:
        """Run the server until cancelled. Cancellation asks uvicorn to
        stop and waits for it, so the tab gets a close frame instead of
        a reset — then re-raises, as a cancelled task should.
        """
        self.bind()
        assert self._server is not None and self._sock is not None
        runner = asyncio.ensure_future(self._server.serve(sockets=[self._sock]))
        try:
            await asyncio.shield(runner)
        except asyncio.CancelledError:
            self._server.should_exit = True
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(runner, SHUTDOWN_TIMEOUT_S * 2)
            raise
        finally:
            self._sock.close()
            self._sock = None
