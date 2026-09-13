"""`neiro talk` — spacebar, speak, spacebar, hear her answer.

Stage 0 Task 10 from one command. Until this existed the daemon, the
orchestrator and every provider were reachable only from tests and
scripts: `neiro say` synthesised a line and `neiro ptt` recorded one,
and nothing joined them. This is the join, kept deliberately thin — the
loop's logic lives in Daemon and Orchestrator, and this module owns
only the three things a terminal needs: a keyboard, a microphone and a
speaker.

**Barge-in is the spacebar.** Pressing it while she is speaking
cancels the turn in flight (`Daemon.interrupt()`), kills playback and
starts recording the correction — the same path the browser's
VAD-driven barge-in takes in Stage 2, exercised by a human today.

**Playback is a child process** (paplay), not a sounddevice output
stream. The loop is single-threaded asyncio, and a PortAudio output
stream would have to stay alive across the await that runs the next
turn; a child process is simply killable mid-sentence, which is all
barge-in asks of it.

**Affect is observed while he speaks.** Every `AFFECT_INTERVAL_S` the
loop hands the daemon the last `window_seconds` from the ring, so the
annotation is ready at the endpoint and costs the turn budget nothing —
the design's central latency trick, working from the first command
that runs the loop.

Everything a test needs to drive this is injectable: keys, capture,
playback and the daemon's providers.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self

import numpy as np

from neiro import metrics
from neiro.config import Neiro
from neiro.daemon import AFFECT_INTERVAL_S, Daemon
from neiro.orchestrator import TurnResult

log = logging.getLogger(__name__)

REPLY_PATH = Path("/tmp/neiro-reply.wav")
# How often the loop looks at the keyboard and the meter. 50 ms is
# imperceptible on a keypress and cheap enough to run forever.
KEY_POLL_S = 0.05
# How long a cancelled turn gets to unwind before the loop gives up
# waiting on it at exit. Generous: STT in its executor cannot be
# interrupted and takes up to ~400 ms.
DRAIN_TIMEOUT_S = 2.0
# Tried in order; the first one present plays the reply.
PLAYERS: tuple[tuple[str, ...], ...] = (("paplay",), ("aplay", "-q"))


class Capture(Protocol):
    """A microphone that keeps a ring buffer and can be armed."""

    @property
    def armed(self) -> bool: ...

    @property
    def level(self) -> float: ...

    def arm(self) -> None: ...

    def disarm(self) -> np.ndarray: ...

    def recent(self, seconds: float) -> np.ndarray: ...


class Playback(Protocol):
    def terminate(self) -> None: ...

    def poll(self) -> int | None: ...


class MicCapture:
    """sounddevice + RingBuffer, the same shape as `neiro ptt`.

    Constructed lazily so importing this module never touches PortAudio;
    the tests drive `TalkLoop` with a fake.
    """

    def __init__(self, cfg: Neiro) -> None:
        import threading

        import sounddevice as sd

        from neiro.audio.devices import (
            apply_to_environment,
            resolve_active_profile,
            suppress_alsa_errors,
        )
        from neiro.audio.ptt import METER_SMOOTHING
        from neiro.audio.ring import RingBuffer

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


@dataclass
class TalkLoop:
    """The keyboard-driven loop. Nothing here knows how a model works."""

    daemon: Daemon
    cfg: Neiro
    capture: Capture
    read_key: Callable[[], str | None]
    play: Callable[[Path], Playback | None]
    out: Callable[[str], None] = print
    on_tick: Callable[[str, float], None] | None = None
    reply_path: Path = REPLY_PATH
    turns_path: Path | None = None  # None: do not append to turns.jsonl

    turns_done: int = 0
    interrupted: int = 0
    results: list[TurnResult] = field(default_factory=list)
    _annotation: str | None = field(default=None, repr=False)
    _inflight: asyncio.Task | None = field(default=None, repr=False)
    _playing: Playback | None = field(default=None, repr=False)

    # -- the loop -------------------------------------------------------

    async def run(self, max_turns: int | None = None) -> int:
        """Poll keys until 'q' (or `max_turns` completed turns, for tests)."""
        last_observe = 0.0
        try:
            while True:
                key = self.read_key()
                if key == "q":
                    return 0
                if key == " ":
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
        from neiro.audio.sink_local import LocalWavSink

        # A fresh sink per turn: the file sink accumulates, and a reply
        # must never start with the tail of the previous one.
        sink = LocalWavSink(path=self.reply_path)
        assert self.daemon.orchestrator is not None, "call build() first"
        self.daemon.orchestrator.sink = sink

        result = await self.daemon.handle(audio, annotation)
        self.turns_done += 1
        self.results.append(result)

        if result.cancelled:
            self.out("(interrupted)")
            return result
        if result.error is not None:
            self.out(f"[{result.error}] {self.daemon.spoken_error(result)}")
            return result

        self.out(f"you    {result.transcript}")
        self.out(f"neiro  {result.reply}")
        record = metrics.record_turn(result.turn)
        self.out(metrics.format_hud(record))
        if self.turns_path is not None:
            metrics.append_turn(record, self.turns_path)

        path = sink.close()
        if path is not None:
            self._playing = self.play(path)
        return result

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


# -- the command --------------------------------------------------------


def _tty_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


def run(cfg: Neiro | None = None) -> int:
    """`neiro talk`. Needs a real terminal, like `neiro ptt`."""
    from rich.console import Console
    from rich.live import Live

    from neiro.audio.ptt import _bar

    cfg = cfg or Neiro()
    console = Console()
    if not sys.stdin.isatty():
        console.print(
            "[red]neiro talk needs a real interactive terminal[/red] (it reads raw "
            "keypresses). Run it directly in a terminal, not piped or backgrounded."
        )
        return 1

    daemon = Daemon(cfg)
    daemon.build()
    console.print("warming…")
    console.print(daemon.warm().describe())
    console.print(
        f"[bold]neiro talk[/bold] — profile '{cfg.audio.active_profile}'. "
        "SPACE to start/stop talking (SPACE while she speaks interrupts her), q to quit.\n"
    )

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
            )
            return asyncio.run(loop.run())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        console.print("\nbye.")
