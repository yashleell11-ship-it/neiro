"""`neiro run` — the daemon that holds the loop together.

Everything else in this package is a component behind a Protocol. This
is the only place that decides *which* ones, and it is deliberately
small: if wiring needs cleverness, the Protocols were wrong.

Three things it owns that nothing else can:

**Warm-up order.** Four runtimes here each cost far more on their first
call than on every subsequent one — ctranslate2 7.44 s (Gate G1),
librosa ~1.21 s, Kokoro several seconds, onnxruntime ~1.5 ms. Warming
them at start is not an optimisation; without it the first thing Yash
says in the morning stalls for ten seconds and the thing looks broken.
They are warmed in the order a turn uses them, so if start-up is
interrupted the parts that ran are the parts needed first.

**`gc.freeze()` after loading.** Every model object loaded at start
lives for the process lifetime, and leaving them in the young generation
means every later collection walks them. Freezing moves them out of
reach of the collector entirely.

**One turn at a time.** A lock, not a queue. If Yash speaks while she is
answering, that is barge-in — a new turn that *cancels* the old one —
not a second turn to run afterwards.

**The confirmation gate is wired here, not in the registry.** The
registry knows a YELLOW tool needs a yes; only the daemon knows what
this machine can ask with. Today that is the clickable notification
alone — the spoken channel needs the microphone and STT that `talk.py`
owns, and until it is joined the click is the one channel that cannot
hallucinate anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from neiro.affect.null import NullAffectProvider
from neiro.affect.prosody import ProsodyAffectProvider, describe
from neiro.config import Neiro
from neiro.orchestrator import Orchestrator, TurnResult
from neiro.speech.errors import ErrorSpeech, Failure
from neiro.state import Turn
from neiro.tools.audit import AuditLog
from neiro.tools.builtin import build_registry
from neiro.tools.confirm import NotificationConfirmer, decide
from neiro.tools.registry import ToolRegistry, ToolSpec

log = logging.getLogger(__name__)

# How often affect re-reads the rolling window while he is speaking.
AFFECT_INTERVAL_S = 0.75


@dataclass
class WarmReport:
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return sum(self.stages.values())

    def describe(self) -> str:
        parts = [f"{name} {seconds * 1000:.0f}ms" for name, seconds in self.stages.items()]
        return f"warm: {', '.join(parts)}  (total {self.total_s:.1f}s)"


class Daemon:
    """Providers in, turns out. Nothing here knows how a model works."""

    def __init__(self, cfg: Neiro | None = None) -> None:
        self.cfg = cfg or Neiro()
        self.orchestrator: Orchestrator | None = None
        self.affect = None
        self._lock = asyncio.Lock()
        # Two turns can exist at once, and they are never the same
        # variable: `_pending` is the utterance HE is making (speech-start
        # to endpoint, while affect watches), `_live` is the turn SHE is
        # answering (inside `run()`). Barge-in cancels `_live` and nothing
        # else. One field for both made `handle()` cancel the very turn
        # it was about to run — every turn with a speaking phase came
        # back empty and `cancelled`, and no test noticed.
        self._pending: Turn | None = None
        self._live: Turn | None = None
        self.errors = ErrorSpeech()
        self.tools: ToolRegistry | None = None
        # The clickable half of the confirmation. A field so a test can
        # hand it a fake runner instead of a desktop.
        self.notifier = NotificationConfirmer()

    # -- start-up -------------------------------------------------------

    def build(
        self,
        stt=None,
        llm=None,
        tts=None,
        sink=None,
        *,
        tools: ToolRegistry | None = None,
        confirm=None,
        audit: AuditLog | None = None,
    ) -> Orchestrator:
        """Choose the providers. Injectable so the daemon is testable.

        The builtin tools are registered here because this is the one
        place that decides what runs; the orchestrator only ever sees a
        registry. `tools` replaces the whole registry (a test's fakes);
        `confirm` and `audit` replace just the gate and the log of the
        real one.
        """
        from neiro.audio.sink_local import LocalWavSink
        from neiro.llm.ollama_native import OllamaNativeLlm
        from neiro.stt.faster_whisper import FasterWhisperStt
        from neiro.tts.kokoro import KokoroTts

        # Affect is off until gate G3b passes on HIS recordings. The null
        # provider is interchangeable with the real one, so turning it on
        # is a config flip rather than a code path.
        self.affect = (
            ProsodyAffectProvider(cfg=self.cfg, device=self.cfg.audio.active_profile)
            if self.cfg.affect.enabled
            else NullAffectProvider()
        )
        self.tools = tools or build_registry(
            confirm=confirm or self.notification_confirm, audit=audit or AuditLog()
        )
        self.orchestrator = Orchestrator(
            stt=stt or FasterWhisperStt(self.cfg),
            llm=llm or OllamaNativeLlm(self.cfg),
            tts=tts or KokoroTts(self.cfg),
            sink=sink or LocalWavSink(),
            affect=self.affect,
            cfg=self.cfg,
            tools=self.tools,
        )
        return self.orchestrator

    # -- confirmation ---------------------------------------------------

    @staticmethod
    def confirmation_question(spec: ToolSpec, parsed: BaseModel) -> str:
        """What the notification asks. Names the tool and every argument,
        because a yes must be a yes to *this* — "set volume, percent 30?"
        — and arguments are enums and integers by construction, so there
        is nothing here the model wrote.
        """
        args = ", ".join(f"{key} {value}" for key, value in parsed.model_dump().items())
        what = spec.name.replace("_", " ")
        return f"{what}, {args}?" if args else f"{what}?"

    def notification_confirm(self, spec: ToolSpec, parsed: BaseModel, nonce: str) -> bool:
        """The registry's gate, over the click channel.

        The nonce is deliberately unused: it binds the registry's own
        record of what was asked, and never appears where the model
        could read it — which includes the notification.
        """
        answer = self.notifier.ask(self.confirmation_question(spec, parsed))
        return decide(None, answer).allowed

    def warm(self) -> WarmReport:
        """Pay every first-call cost now, in the order a turn uses them.

        If start-up is interrupted, the parts that ran are the parts
        needed first — which is why this is not a `gather()`.
        """
        report = WarmReport()
        assert self.orchestrator is not None, "call build() first"

        for name, target in (
            ("affect", self.affect),
            ("stt", self.orchestrator.stt),
            ("tts", self.orchestrator.tts),
        ):
            warm = getattr(target, "warm", None)
            if warm is None:
                continue
            started = time.perf_counter()
            try:
                warm()
            except Exception as exc:  # noqa: BLE001 — a cold component still works, slowly
                log.warning("warming %s failed (%s); first turn will be slow", name, exc)
            report.stages[name] = time.perf_counter() - started

        # Everything above lives for the process lifetime. Freezing moves
        # it out of the collector's reach instead of walking it forever.
        gc.collect()
        gc.freeze()
        return report

    # -- turns ----------------------------------------------------------

    def begin_utterance(self) -> Turn:
        """He has started speaking. Creates the Turn that `observe()`
        fills and `handle()` runs at the endpoint.

        Separate from `handle()` because affect observes a rolling
        window *while he is still speaking*; the Turn has to exist then
        or the reading has nowhere to go.
        """
        assert self.orchestrator is not None, "call build() first"
        self._pending = self.orchestrator.begin_turn()
        return self._pending

    async def observe(self, window: np.ndarray) -> str | None:
        """Feed the affect provider a rolling window while he speaks.

        Returns the annotation the prompt would receive, or None. Costs
        the turn budget nothing because it happens before the endpoint.
        """
        if self.orchestrator is None:
            return None
        turn = self._pending or self.begin_utterance()
        affect = await self.orchestrator.observe_while_speaking(turn, window)
        return describe(affect, self.cfg)

    async def handle(self, audio: np.ndarray, annotation: str | None = None) -> TurnResult:
        """One utterance. Barge-in cancels the turn in flight first."""
        assert self.orchestrator is not None, "call build() first"
        # Claim his turn before waiting for the lock: windows keep
        # arriving while the cancelled turn gets out of the way, and the
        # utterance AFTER this one must not be what runs on this audio.
        turn = self._pending or self.orchestrator.begin_turn()
        self._pending = None
        await self.interrupt()
        async with self._lock:
            self._live = turn
            try:
                result = await self.orchestrator.run(audio, annotation, turn=turn)
            finally:
                # Cleared in a `finally` so a turn torn down mid-flight
                # cannot leave a stale Turn that the next barge-in would
                # cancel instead of the real one.
                self._live = None
        return result

    async def interrupt(self) -> bool:
        """Barge-in. Returns True if a turn was actually cancelled.

        Only the turn in flight — the one she is answering. The
        utterance he is in the middle of is not something to interrupt;
        nothing is playing.

        Setting the event is enough: every stage checks it at its awaits,
        which is why it had to exist from the first line rather than be
        added when barge-in landed.
        """
        turn = self._live
        if turn is None or turn.cancel.is_set():
            return False
        turn.cancel.set()
        return True

    def spoken_error(self, result: TurnResult) -> str | None:
        """What she says when a turn failed. In character, and short."""
        if result.error is None:
            return None
        failure = {
            "empty_transcript": Failure.NOT_HEARD,
            "ConnectionError": Failure.LLM_DOWN,
            "ThinkingModeError": Failure.LLM_DOWN,
            "ReadTimeout": Failure.LLM_SLOW,
            "TimeoutException": Failure.LLM_SLOW,
            # He said no, or nothing, to a YELLOW tool. Not a failure of
            # hers, and "that didn't work" would claim she tried.
            "ToolNotConfirmed": Failure.TOOL_DENIED,
        }.get(result.error, Failure.TOOL_FAILED)
        return self.errors.line_for(failure)

    async def aclose(self) -> None:
        await self.interrupt()
        if self.affect is not None and hasattr(self.affect, "save"):
            # The prosody baseline is the only state worth persisting —
            # without it she is emotionally blind for the first minute of
            # every day.
            with contextlib.suppress(OSError):
                self.affect.save()
