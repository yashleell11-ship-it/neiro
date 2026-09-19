"""`elizabeth talk` — say her name, speak, hear her answer.

Stage 0 Task 10 from one command. Until this existed the daemon, the
orchestrator and every provider were reachable only from tests and
scripts: `elizabeth say` synthesised a line and `elizabeth ptt` recorded one,
and nothing joined them. This is the join, kept deliberately thin — the
loop's logic lives in Daemon and Orchestrator, and this module owns
only the three things a terminal needs: a keyboard, a microphone and a
speaker.

**The wake word presses the spacebar for you.** `wake.py` scores the mic
continuously and `toggle()` runs on the hop that crosses, which is the
same path the key takes — so hands-free listening adds no second way for
a turn to start, and the key still works when the head is untrained, the
model file is missing, or onnxruntime will not load. Each of those
prints a line and leaves you with the spacebar; none of them is fatal.
The mic queue is drained every tick either way, and the detector's
context is thrown away rather than scored whenever the mic is armed or a
block was dropped, because a gap makes the last two seconds a lie.

**Barge-in is the spacebar.** Pressing it while she is speaking
cancels the turn in flight (`Daemon.interrupt()`), kills playback and
starts recording the correction — the same path the browser's
VAD-driven barge-in takes in Stage 2, exercised by a human today.

**Playback is a child process** (paplay), not a sounddevice output
stream. The loop is single-threaded asyncio, and a PortAudio output
stream would have to stay alive across the await that runs the next
turn; a child process is simply killable mid-sentence, which is all
barge-in asks of it.

**`--browser` swaps the speaker for the face.** The sink becomes the
WebSocket one, the server runs as a task beside the loop on the same
event loop, and the reply is played by the tab — which is the only
place the one metric can honestly end, because only the tab knows when
sound left the speaker. The loop does not change shape for it: a
`sink_factory` decides what a turn plays through, and the finished
sink stands in for the paplay process so SPACE during the tail of a
reply stops the browser the way it kills the player.

**Affect is observed while he speaks.** Every `AFFECT_INTERVAL_S` the
loop hands the daemon the last `window_seconds` from the ring, so the
annotation is ready at the endpoint and costs the turn budget nothing —
the design's central latency trick, working from the first command
that runs the loop.

Everything a test needs to drive this is injectable: keys, capture,
playback, the sink, the server and the daemon's providers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import select
import subprocess
import sys
import termios
import time
import tty
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np

from elizabeth import metrics
from elizabeth.audio.sink_ws import FIRST_AUDIO_TIMEOUT_S
from elizabeth.audio.wake import WakeModelMissing, WakeWord
from elizabeth.config import Elizabeth
from elizabeth.daemon import AFFECT_INTERVAL_S, Daemon
from elizabeth.orchestrator import TurnResult
from elizabeth.protocols import Sink

if TYPE_CHECKING:
    from elizabeth.server import Face

log = logging.getLogger(__name__)

REPLY_PATH = Path("/tmp/elizabeth-reply.wav")
# How often the loop looks at the keyboard and the meter. 50 ms is
# imperceptible on a keypress and cheap enough to run forever.
KEY_POLL_S = 0.05
# How long a cancelled turn gets to unwind before the loop gives up
# waiting on it at exit. Generous: STT in its executor cannot be
# interrupted and takes up to ~400 ms.
DRAIN_TIMEOUT_S = 2.0
# How much unexamined audio the wake word's queue may hold before the
# oldest is dropped. Two seconds is more than a 50 ms poll ever needs;
# reaching it means the loop stalled, and the detector is told so rather
# than being fed a stream with a hole in it.
MONITOR_MAX_SECONDS = 2.0
# Tried in order; the first one present plays the reply.
PLAYERS: tuple[tuple[str, ...], ...] = (("paplay",), ("aplay", "-q"))
# The command that opens the face. One attempt, never retried: a desktop
# without it prints the URL and that is enough.
OPENERS: tuple[tuple[str, ...], ...] = (("xdg-open",),)


class Capture(Protocol):
    """A microphone that keeps a ring buffer and can be armed."""

    @property
    def armed(self) -> bool: ...

    @property
    def level(self) -> float: ...

    def arm(self) -> None: ...

    def disarm(self) -> np.ndarray: ...

    def recent(self, seconds: float) -> np.ndarray: ...

    def monitor(self) -> tuple[np.ndarray, bool]: ...


@runtime_checkable
class Playback(Protocol):
    """Something that is speaking right now and can be stopped: a paplay
    process, or the browser sink of a turn that has finished sending."""

    def terminate(self) -> None: ...

    def poll(self) -> int | None: ...


@runtime_checkable
class ReportsFirstAudio(Protocol):
    """A sink whose first-audio moment arrives later, from somewhere else
    — the browser's `played`. The HUD waits for it, bounded."""

    async def wait_played(self, timeout: float) -> bool: ...


class MicCapture:
    """sounddevice + RingBuffer, the same shape as `elizabeth ptt`.

    Constructed lazily so importing this module never touches PortAudio;
    the tests drive `TalkLoop` with a fake.
    """

    def __init__(self, cfg: Elizabeth) -> None:
        import threading

        import sounddevice as sd

        from elizabeth.audio.devices import (
            apply_to_environment,
            resolve_active_profile,
            suppress_alsa_errors,
        )
        from elizabeth.audio.ptt import METER_SMOOTHING
        from elizabeth.audio.ring import RingBuffer

        suppress_alsa_errors()
        device = resolve_active_profile(cfg)
        apply_to_environment(device)
        self.samplerate = cfg.audio.input_samplerate
        self._pre_roll_s = cfg.audio.pre_roll_s
        self._smoothing = METER_SMOOTHING
        # The ring must hold the affect window, not just the pre-roll.
        self._ring = RingBuffer(
            capacity_seconds=max(cfg.audio.ring_buffer_seconds, cfg.affect.window_seconds),
            samplerate=self.samplerate,
        )
        self._lock = threading.Lock()
        self._armed = False
        self._level = 0.0
        self._utterance: list[np.ndarray] = []
        # Blocks the wake word has not looked at yet. Appended in the
        # PortAudio callback and drained by the loop — never processed in
        # the callback itself, which has a hard deadline and would drop
        # audio outright if a 1.75 ms ONNX call ran inside it.
        self._monitor: list[np.ndarray] = []
        self._monitor_dropped = False
        self._monitor_max_blocks = max(
            1, int(MONITOR_MAX_SECONDS * self.samplerate / cfg.audio.input_blocksize)
        )
        self._stream = sd.InputStream(
            device=device.portaudio_device,
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=cfg.audio.input_blocksize,
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status) -> None:
        chunk = indata[:, 0].copy()
        self._ring.write(chunk)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        with self._lock:
            self._level = (1 - self._smoothing) * self._level + self._smoothing * rms
            if self._armed:
                self._utterance.append(chunk)
            self._monitor.append(chunk)
            if len(self._monitor) > self._monitor_max_blocks:
                # The consumer stalled. Dropping the oldest block is right —
                # a wake detector fed across a gap is scoring a sentence
                # that was never spoken — but it must be ADMITTED, so the
                # loop resets the context instead of trusting it.
                del self._monitor[0]
                self._monitor_dropped = True

    def __enter__(self) -> Self:
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stream.stop()
        self._stream.close()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def level(self) -> float:
        with self._lock:
            return self._level

    def arm(self) -> None:
        with self._lock:
            preroll = self._ring.read_last(self._pre_roll_s)
            self._utterance = [preroll] if preroll.size else []
            self._armed = True

    def disarm(self) -> np.ndarray:
        with self._lock:
            self._armed = False
            chunks, self._utterance = self._utterance, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def recent(self, seconds: float) -> np.ndarray:
        return self._ring.read_last(seconds)

    def monitor(self) -> tuple[np.ndarray, bool]:
        """Everything captured since the last call, and whether a gap opened.

        Separate from `recent()` on purpose: `recent` reads a WINDOW out
        of the ring and will happily hand back the same audio twice or
        skip a slice, which is fine for an affect window averaged over
        three seconds and wrong for a detector whose whole input is an
        ordered, gapless stream.
        """
        with self._lock:
            chunks, self._monitor = self._monitor, []
            dropped, self._monitor_dropped = self._monitor_dropped, False
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, dropped


def subprocess_player(path: Path) -> Playback | None:
    """Start playing `path`; return the process, or None if no player exists."""
    for command in PLAYERS:
        try:
            return subprocess.Popen(
                [*command, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            continue
    log.warning("no audio player found (tried %s); reply written to %s", PLAYERS, path)
    return None


def xdg_open(url: str) -> bool:
    """Open `url` in whatever the desktop calls a browser. False if there
    is no opener; the URL was printed, and that is the fallback.
    """
    for command in OPENERS:
        try:
            subprocess.Popen([*command, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
        return True
    return False


@dataclass
class TalkLoop:
    """The keyboard-driven loop. Nothing here knows how a model works."""

    daemon: Daemon
    cfg: Elizabeth
    capture: Capture
    read_key: Callable[[], str | None]
    play: Callable[[Path], Playback | None]
    out: Callable[[str], None] = print
    on_tick: Callable[[str, float], None] | None = None
    reply_path: Path = REPLY_PATH
    turns_path: Path | None = None  # None: do not append to turns.jsonl
    # What each turn plays through. None: a fresh LocalWavSink per turn,
    # played with `play`. `--browser` supplies WebSocket sinks instead.
    sink_factory: Callable[[], Sink] | None = None
    # Runs beside the loop for its whole life — the face's server. Started
    # before the first key is read, cancelled when the loop ends; if it
    # dies first, the loop ends with an error rather than talking to
    # nobody.
    server: Callable[[], Awaitable[None]] | None = None
    # The always-on wake word. None keeps the loop exactly as it was:
    # spacebar only. Set, it presses the spacebar for you.
    wake: WakeWord | None = None

    turns_done: int = 0
    woke: int = 0
    interrupted: int = 0
    results: list[TurnResult] = field(default_factory=list)
    _annotation: str | None = field(default=None, repr=False)
    _inflight: asyncio.Task | None = field(default=None, repr=False)
    _playing: Playback | None = field(default=None, repr=False)
    _server_task: asyncio.Task | None = field(default=None, repr=False)

    # -- the loop -------------------------------------------------------

    async def run(self, max_turns: int | None = None) -> int:
        """Poll keys until 'q' (or `max_turns` completed turns, for tests)."""
        last_observe = 0.0
        if self.server is not None:
            self._server_task = asyncio.create_task(self.server())
            # Let it actually start before the first key is read: a task
            # only runs at the next await, and the tab may already be
            # trying to connect.
            await asyncio.sleep(0)
        try:
            while True:
                if self._server_task is not None and self._server_task.done():
                    exc = self._server_task.exception()
                    self.out(f"face server stopped: {exc or 'exited'}")
                    return 1
                key = self.read_key()
                if key == "q":
                    return 0
                if key == " ":
                    await self.toggle()
                elif await self._heard_her_name():
                    await self.toggle()

                if self.capture.armed:
                    now = time.monotonic()
                    if now - last_observe >= AFFECT_INTERVAL_S:
                        last_observe = now
                        window = self.capture.recent(self.cfg.affect.window_seconds)
                        self._annotation = await self.daemon.observe(window)

                self._reap()
                if self.on_tick is not None:
                    self.on_tick(self.phase, self.capture.level)
                if (
                    max_turns is not None
                    and self.turns_done >= max_turns
                    and self._inflight is None
                ):
                    return 0
                await asyncio.sleep(KEY_POLL_S)
        finally:
            await self._stop_everything()
            await self.daemon.aclose()
            if self._server_task is not None and not self._server_task.done():
                self._server_task.cancel()
                await asyncio.gather(self._server_task, return_exceptions=True)

    async def _heard_her_name(self) -> bool:
        """Drain the mic into the wake word; True when it fired.

        The queue is drained on EVERY tick, wake word or not, so a run
        with the detector off does not slowly fill a list nobody reads.
        While the mic is armed the context is discarded rather than
        scored: she is already recording, and the audio she is recording
        is the question, not the name.
        """
        if not hasattr(self.capture, "monitor"):
            return False
        audio, dropped = self.capture.monitor()
        if self.wake is None:
            return False
        if self.capture.armed or dropped:
            # A gap, or a state change. Either makes the last two seconds
            # of context a lie, and a lie is worse than starting over.
            self.wake.reset()
            return False
        if not audio.size:
            return False
        event = self.wake.feed(audio, now=time.monotonic())
        if event is None:
            return False
        self.woke += 1
        self.out(f"— heard you ({event.score:.2f})")
        return True

    @property
    def phase(self) -> str:
        if self.capture.armed:
            return "recording"
        if self._inflight is not None and not self._inflight.done():
            return "thinking"
        if self._playing is not None and self._playing.poll() is None:
            return "speaking"
        return "listening"

    # -- one keypress ---------------------------------------------------

    async def toggle(self) -> None:
        """SPACE. Arms the mic, or ends the utterance and runs the turn."""
        if self.capture.armed:
            audio = self.capture.disarm()
            annotation, self._annotation = self._annotation, None
            self._inflight = asyncio.create_task(self._turn(audio, annotation))
            return

        # Arming while she is mid-reply is barge-in: the turn in flight
        # is cancelled and playback dies before the mic opens.
        mid_reply = self._inflight is not None and not self._inflight.done()
        if mid_reply and await self.daemon.interrupt():
            self.interrupted += 1
        self._kill_playback()
        self.daemon.begin_utterance()
        self.capture.arm()

    # -- one turn -------------------------------------------------------

    async def _turn(self, audio: np.ndarray, annotation: str | None) -> TurnResult:
        # A fresh sink per turn: a sink accumulates (a file, an audio_id),
        # and a reply must never start with the tail of the previous one.
        sink = self._new_sink()
        assert self.daemon.orchestrator is not None, "call build() first"
        self.daemon.orchestrator.sink = sink

        result = await self.daemon.handle(audio, annotation)
        self.turns_done += 1
        self.results.append(result)

        if result.cancelled:
            self.out("(interrupted)")
            return result
        if result.error is not None:
            # Closed, not played: the browser gets its `utt.end` for
            # whatever was sent before the failure, and nothing starts.
            _close(sink)
            self.out(f"[{result.error}] {self.daemon.spoken_error(result)}")
            return result

        self.out(f"you    {result.transcript}")
        self.out(f"elizabeth  {result.reply}")
        self._playing = self._start_playback(sink)
        if isinstance(sink, ReportsFirstAudio):
            # The metric's end is the browser's report, which arrives on
            # its own time. Bounded, so a tab that never answers costs a
            # HUD line that says "no audio" rather than a hung loop.
            await sink.wait_played(FIRST_AUDIO_TIMEOUT_S)
        record = metrics.record_turn(result.turn)
        self.out(metrics.format_hud(record))
        if self.turns_path is not None:
            metrics.append_turn(record, self.turns_path)
        return result

    def _new_sink(self) -> Sink:
        if self.sink_factory is not None:
            return self.sink_factory()
        from elizabeth.audio.sink_local import LocalWavSink

        return LocalWavSink(path=self.reply_path)

    def _start_playback(self, sink: Sink) -> Playback | None:
        """A file sink hands back a path the player plays; the browser
        sink played it already and hands back itself, so `phase` and
        barge-in treat the tail of the reply the same way either way.
        """
        path = _close(sink)
        if path is not None:
            return self.play(path)
        return sink if isinstance(sink, Playback) else None

    # -- housekeeping ---------------------------------------------------

    def _reap(self) -> None:
        if self._inflight is not None and self._inflight.done():
            exc = self._inflight.exception() if not self._inflight.cancelled() else None
            if exc is not None:
                log.error("turn failed: %s", exc)
                self.out(f"turn failed: {exc}")
            self._inflight = None
        if self._playing is not None and self._playing.poll() is not None:
            self._playing = None

    def _kill_playback(self) -> None:
        if self._playing is not None:
            with contextlib.suppress(Exception):
                self._playing.terminate()
            self._playing = None

    async def _stop_everything(self) -> None:
        await self.daemon.interrupt()
        self._kill_playback()
        if self._inflight is not None and not self._inflight.done():
            with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._inflight, DRAIN_TIMEOUT_S)
        self._inflight = None


def _close(sink: Sink) -> Path | None:
    """`close()` is the sinks' own end-of-reply call, not part of the
    `Sink` protocol the orchestrator sees — the file sink writes its
    WAV and returns the path, the browser sink sends `utt.end` and
    returns None. A sink without one has nothing to finish.
    """
    close = getattr(sink, "close", None)
    return close() if close is not None else None


# -- the command --------------------------------------------------------


def _tty_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


def browser_options(
    cfg: Elizabeth, out: Callable[[str], None] = print, port: int | None = None
) -> tuple[Face, dict]:
    """The `TalkLoop` fields `--browser` sets, and the face they belong to.

    The port is bound here, before any model loads: a taken port is an
    `OSError` with a message at the top of the run, not a stopped server
    task twenty seconds of warm-up later. `port` None is the server's
    default; tests pass 0 for a free one.
    """
    from elizabeth.audio.sink_ws import WsSink
    from elizabeth.server import DEFAULT_PORT, Face

    face = Face(cfg, port=DEFAULT_PORT if port is None else port)
    out(f"face: {face.bind()}")
    options = {
        "sink_factory": lambda: WsSink(face.session, cfg),
        "server": face.serve,
    }
    return face, options


def run(cfg: Elizabeth | None = None, browser: bool = False) -> int:
    """`elizabeth talk`. Needs a real terminal, like `elizabeth ptt`."""
    from rich.console import Console
    from rich.live import Live

    from elizabeth.audio.ptt import _bar

    cfg = cfg or Elizabeth()
    console = Console()
    if not sys.stdin.isatty():
        console.print(
            "[red]elizabeth talk needs a real interactive terminal[/red] (it reads raw "
            "keypresses). Run it directly in a terminal, not piped or backgrounded."
        )
        return 1

    face, options = None, {}
    if browser:
        try:
            face, options = browser_options(cfg, out=console.print)
        except OSError as exc:
            console.print(f"[red]cannot serve the face:[/red] {exc}")
            return 1

    daemon = Daemon(cfg)
    daemon.build()
    console.print("warming…")
    console.print(daemon.warm().describe())

    # The wake word is a NICE-TO-HAVE on top of the spacebar, never a
    # replacement for it at startup: an untrained head, a missing model
    # file or a broken onnxruntime must cost you hands-free listening,
    # not the ability to talk to her at all.
    wake = None
    if cfg.wake.enabled:
        try:
            wake = WakeWord.with_models(cfg)
        except WakeModelMissing as exc:
            console.print(f"[yellow]wake word off:[/yellow] {exc}")
        except Exception as exc:  # noqa: BLE001 - never fatal, see above
            console.print(f"[yellow]wake word off:[/yellow] {type(exc).__name__}: {exc}")

    hands_free = f'say "{cfg.wake.phrase}" or ' if wake is not None else ""
    console.print(
        f"[bold]elizabeth talk[/bold] — profile '{cfg.audio.active_profile}'. "
        f"{hands_free}SPACE to start/stop talking "
        "(SPACE while she speaks interrupts her), q to quit.\n"
    )
    if face is not None and not xdg_open(face.url):
        console.print(f"open {face.url} in a browser.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    labels = {
        "listening": "listening",
        "recording": "[red]● RECORDING[/red]",
        "thinking": "[yellow]thinking…[/yellow]",
        "speaking": "[green]speaking[/green]",
    }
    try:
        tty.setraw(fd)
        with (
            MicCapture(cfg) as capture,
            Live(console=console, refresh_per_second=20, transient=False) as live,
        ):
            loop = TalkLoop(
                daemon=daemon,
                cfg=cfg,
                capture=capture,
                read_key=_tty_key,
                play=subprocess_player,
                out=lambda line: console.print(f"\n{line}"),
                on_tick=lambda phase, level: live.update(f"{_bar(level)}  {labels[phase]}"),
                turns_path=metrics.TURNS_PATH,
                wake=wake,
                **options,
            )
            return asyncio.run(loop.run())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        console.print("\nbye.")
