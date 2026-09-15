"""The sink that feeds the browser: her voice, and the face that moves with it.

The orchestrator calls `play()` once per synthesised chunk and `cancel()`
on barge-in — the same two calls `LocalWavSink` gets, so it cannot tell
the file from the browser. What this one does with them is push the
protocol frozen in `server.py`:

    utt.begin   on the first chunk — her emotion, and how fast the face
                should arrive at it (the blender's rise time, from config)
    utt.chunk   + one binary audio frame, per chunk — seq, duration, the
                text, and the viseme timeline the mouth is animated from
    emotion     only if her state changed between chunks; the tag resolves
                before the first sentence so this is rare, and sending it
                every chunk would make the face twitch
    utt.end     from `close()`, which the talk loop calls once the turn
                is over — the orchestrator has no "end" call, deliberately
    cancel      from `cancel()`, after everything still queued is dropped

**Nothing here waits forever.** The queue to the socket is bounded and a
tab that stops draining it — a background tab Chrome has throttled, a
laptop lid closing — must not park the orchestrator on a `put()` that
never returns: the daemon holds its lock across the turn, so that would
be the whole assistant dead until restart, the exact shape of the deadlock
in `git log` (2296ba0). So `play()` waits a bounded time for room and
then *drops the chunk and says so*. A dropped chunk is a glitch he hears
once; a stalled pipeline is a restart.

A chunk is dropped whole. Its header and its PCM are one queue entry
(`server.Outgoing`), so they are queued together or not at all — the
browser pairs each binary frame with the header before it, and a header
whose PCM was dropped would be paired with the *next* chunk's audio,
after which every chunk of the reply plays with the wrong seq, text and
visemes. One entry, one slot: there is no half-queued chunk to get wrong.

**`played` is not sent, it is received.** The one metric ends when the
browser reports sequence 0 actually playing, and that report arrives on
the socket after `play()` has long returned. The session stamps the Turn
when it lands; `wait_played()` is how the talk loop gives the HUD a
bounded chance to show the real number instead of printing before it
exists.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from elizabeth.config import Elizabeth
from elizabeth.emotion.blend import ExpressionBlender
from elizabeth.evals.latency import HEADLINE_END
from elizabeth.server import Outgoing, audio_frame, server_message
from elizabeth.state import ElizabethState, Turn

log = logging.getLogger(__name__)

# How long `play()` waits for room in a full outgoing queue before
# dropping the frame. Half a second is longer than any GC pause or tab
# repaint and far shorter than the LLM's HTTP timeout — the bound that
# used to apply when a full queue was waited on without limit.
SEND_TIMEOUT_S = 0.5

# How long the browser is given, once the reply has been sent, to report
# the first chunk playing. The HUD waits this long before printing the
# turn without the number, and `poll()` stops saying "speaking" after it:
# a tab whose AudioContext is still suspended (the unlock button never
# clicked) never reports, and a phase that never comes back would be
# carried by the talk loop for the rest of the run. By the time the wait
# starts the whole reply has been synthesised and sequence 0 was sent
# long ago, so the report is normally already in; the bound is for a tab
# mid-reload, or one that is never going to play.
FIRST_AUDIO_TIMEOUT_S = 2.0

# Viseme times are seconds relative to the chunk start; four decimals is
# a tenth of a millisecond, finer than any frame the face will draw.
VISEME_DECIMALS = 4


class ReplyChannel(Protocol):
    """What the sink needs from the session: the queue, whether a tab is
    on the other end of it, what that tab's avatar can express, and a
    way to get sequence 0's `played` back onto the Turn. `server.Session`
    is the real one; tests hand in something with the same four shapes.
    """

    connected: bool
    expressions: frozenset[str]
    outgoing: asyncio.Queue

    def begin_reply(self, turn: Turn) -> int: ...

    def discard_queued(self) -> int: ...

    def push(self, item: Outgoing) -> bool: ...

    async def wait_played(self, audio_id: int, timeout: float) -> bool: ...


@dataclass
class WsSink:
    """One reply's worth of frames to one browser tab. Fresh per turn, like
    the file sink: an audio_id is minted on the first chunk and every
    frame after it carries that id, so the browser can drop a late chunk
    from a reply that has since been cancelled.
    """

    session: ReplyChannel
    cfg: Elizabeth = field(default_factory=Elizabeth)
    audio_id: int | None = None
    sent: int = 0  # entries that reached the queue; a chunk (header + PCM) is one
    dropped: int = 0  # entries that did not: no tab, or a tab too far behind
    cancelled: bool = False
    ended: bool = False
    _turn: Turn | None = field(default=None, repr=False)
    _state: ElizabethState | None = field(default=None, repr=False)
    _seconds_sent: float = field(default=0.0, repr=False)
    _closed_at: float | None = field(default=None, repr=False)

    # -- the Sink protocol ----------------------------------------------

    async def play(
        self,
        turn: Turn,
        pcm: np.ndarray,
        seq: int = 0,
        text: str = "",
        visemes: list[tuple[float, str, float]] | None = None,
    ) -> None:
        state = turn.elizabeth_state
        if self.audio_id is None:
            self._turn = turn
            self.audio_id = self.session.begin_reply(turn)
            self._state = state
            await self._send(
                server_message("utt.begin", audio_id=self.audio_id, **self._emotion(state)),
                "utt.begin",
            )
        elif state != self._state:
            # Her state moved mid-reply: the face follows, the audio
            # already queued does not change.
            self._state = state
            await self._send(
                server_message("emotion", audio_id=self.audio_id, **self._emotion(state)),
                "emotion",
            )

        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        samplerate = self.cfg.audio.output_samplerate
        seconds = samples.size / samplerate
        timeline = [
            [round(float(start), VISEME_DECIMALS), shape, round(float(held), VISEME_DECIMALS)]
            for start, shape, held in (visemes or [])
        ]
        header = server_message(
            "utt.chunk",
            seq=seq,
            audio_id=self.audio_id,
            dur_ms=round(seconds * 1000),
            text=text,
            visemes=timeline,
        )
        # The header and the PCM are ONE queue entry: the browser matches
        # each binary frame to the header before it, so the two must be
        # queued together or dropped together. Queued one after the
        # other, a header could fit where its PCM then did not, and the
        # browser would pair that header with the NEXT chunk's audio —
        # every chunk after it playing with the wrong seq, text and
        # visemes. One entry takes one slot, so there is no such gap.
        if await self._send((header, audio_frame(self.audio_id, samples)), f"chunk {seq}"):
            self._seconds_sent += seconds

    async def cancel(self, turn: Turn) -> None:
        """Barge-in: everything queued is dropped, then the browser is told
        to stop what it already has. Dropping first is what guarantees
        the `cancel` frame has room — a full queue is exactly when a
        cancel matters most.
        """
        self._cancel_now()

    # -- the talk loop's side -------------------------------------------

    def close(self) -> None:
        """The reply is over. `utt.end` closes the utterance for the face;
        after a cancel there is nothing to end. Returns None, where the
        file sink returns a path: there is no file, the browser played it.
        """
        if self.audio_id is None or self.cancelled or self.ended:
            return
        self.ended = True
        self._closed_at = time.perf_counter()
        if not self.session.push(server_message("utt.end", audio_id=self.audio_id)):
            self.dropped += 1

    async def wait_played(self, timeout: float) -> bool:
        """Until the browser reports sequence 0 playing — the end of the
        one metric — or `timeout`. False is honest: no number, no HUD line
        pretending there is one.
        """
        if self.audio_id is None or self.cancelled:
            return False
        return await self.session.wait_played(self.audio_id, timeout)

    # `Playback`, for the talk loop: it treats the sink of a finished turn
    # the way it treats a paplay process, so SPACE during the tail of a
    # reply stops the browser the same way it kills the player.

    def terminate(self) -> None:
        self._cancel_now()

    def poll(self) -> int | None:
        """None while she is still speaking in the browser, 0 once done.

        The browser reports when playback *started*, not when it ended,
        so the end is the start plus what was sent — an estimate, and
        only the HUD label rests on it.

        Until the browser reports, "speaking" is a guess, and a bounded
        one: a tab that has the frames but never reports (its
        AudioContext still suspended, the unlock button never clicked)
        gets `FIRST_AUDIO_TIMEOUT_S` after `close()` — the same
        allowance the HUD gives the report — and is then given up on.
        Otherwise the loop would carry a reply nobody heard as "speaking"
        until the next keypress, however long that is.
        """
        if self.audio_id is None or self.cancelled or self._turn is None:
            return 0
        if not self.session.connected:
            return 0  # whatever was playing died with the socket
        now = time.perf_counter()
        started = self._turn.timeline.get(HEADLINE_END)
        if started is not None:
            return None if now < started + self._seconds_sent else 0
        if self._closed_at is None:
            return None  # still being sent
        return None if now < self._closed_at + FIRST_AUDIO_TIMEOUT_S else 0

    # -- internals --------------------------------------------------------

    def _emotion(self, state: ElizabethState) -> dict:
        """The fields `utt.begin` and `emotion` share.

        `weights` are the blender's targets — one preset at the tag's
        intensity, redistributed to what the avatar actually has (the
        page told us in `ready`), so an avatar without `surprised` gets
        `happy` rather than nothing. The browser eases toward them on its
        own frame clock with the time constants `hello` gave it.
        """
        blender = ExpressionBlender(cfg=self.cfg, available=self.session.expressions or None)
        return {
            "emotion": state.label.value,
            "intensity": state.intensity,
            "blend_ms": round(self.cfg.expression.tau_rise_s * 1000),
            "weights": blender.target_weights(state),
        }

    def _cancel_now(self) -> None:
        if self.cancelled:
            return
        self.cancelled = True
        if self.audio_id is None:
            return  # nothing was sent; the browser has nothing to stop
        self.session.discard_queued()
        if not self.session.push(server_message("cancel", audio_id=self.audio_id)):
            self.dropped += 1

    async def _send(self, item: Outgoing, what: str) -> bool:
        """Queue one entry — a message, or a chunk's header and PCM as one
        — waiting a bounded time for room. False means dropped, and
        logged with `what` it was, because a silently dropped chunk is
        the kind of bug that gets blamed on the TTS.
        """
        if not self.session.connected:
            self.dropped += 1
            if self.dropped == 1:
                log.warning("no browser tab is connected; this reply is not being played")
            return False
        try:
            self.session.outgoing.put_nowait(item)
        except asyncio.QueueFull:
            try:
                await asyncio.wait_for(self.session.outgoing.put(item), SEND_TIMEOUT_S)
            except TimeoutError:
                self.dropped += 1
                log.warning(
                    "browser tab is not draining audio (waited %.1fs); dropped %s of reply %s",
                    SEND_TIMEOUT_S,
                    what,
                    self.audio_id,
                )
                return False
        self.sent += 1
        return True
