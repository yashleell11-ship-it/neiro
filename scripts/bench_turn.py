#!/usr/bin/env python3
"""Replay fixture turns through fakes and report the one metric.

    uv run scripts/bench_turn.py
    uv run scripts/bench_turn.py --assert       # fail if p50 is over budget

**No mic, no model, no human.** Every provider is a fake with a
configurable delay, so this measures the PIPELINE — queueing,
backpressure, cancellation, the stamp order — rather than how fast
anyone's GPU is today. That is the only way a latency regression test
can run in CI and mean anything.

The delays come from `docs/BUDGET.md`'s measured rows, so a change in
the *code* shows up here while a change in the *hardware* does not. When
the real numbers move, the budget moves and this follows; the two are
deliberately not the same file.

`--assert` is what `just bench` runs before a commit: if the pipeline
itself has gained 200 ms of overhead, that is a bug, and it should be
caught by a test rather than noticed in a demo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from elizabeth.evals.latency import percentile, stage_breakdown
from elizabeth.llm.openai_compat import StreamAccumulator
from elizabeth.orchestrator import Orchestrator

# Measured on this machine; see docs/DECISIONS.md. These are the costs
# the pipeline has to absorb, not costs it creates.
STT_MS = 141.0
LLM_FIRST_TOKEN_MS = 120.0
LLM_PER_CHUNK_MS = 40.0
TTS_FIRST_MS = 384.0  # a 3-word opener, per gate G6
TTS_PER_CHUNK_MS = 120.0

# Pipeline overhead budget: everything the code adds on top of the model
# costs above. Generous, because it is a regression bar rather than a
# target — if this trips, something structural changed.
MAX_OVERHEAD_MS = 60.0

REPLIES = [
    "<e:neutral:4> Ninety six percent. Still charging.",
    "<e:happy:7> Oh, that actually worked? Nice.",
    "<e:relaxed:5> Yeah, it's running. Nothing on fire.",
    "<e:surprised:8> Wait, you already pushed it?",
    "<e:sad:3> Yeah, that happens. What broke?",
]


class FakeStt:
    async def transcribe(self, pcm: np.ndarray) -> str:
        await asyncio.sleep(STT_MS / 1000)
        return "what's my battery at"


class FakeLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def stream(self, messages, tools=None):
        await asyncio.sleep(LLM_FIRST_TOKEN_MS / 1000)
        # Fragment it the way a real token stream arrives, so the chunker
        # and the queue are exercised rather than handed a whole string.
        for i in range(0, len(self.reply), 6):
            await asyncio.sleep(LLM_PER_CHUNK_MS / 1000 / 6)
            yield {"text": self.reply[i : i + 6]}
        yield {"done": StreamAccumulator(text=self.reply)}


class FakeTts:
    def __init__(self) -> None:
        self.calls = 0

    async def synth(self, text: str, state=None):
        first = self.calls == 0
        self.calls += 1
        await asyncio.sleep((TTS_FIRST_MS if first else TTS_PER_CHUNK_MS) / 1000)
        yield np.zeros(240, dtype=np.float32), None


class FakeSink:
    """Stamps `sink_played` so the headline metric is computable.

    A real browser reports this from its AudioContext; a fake has to
    stand in for that moment, and it is named identically on purpose —
    unlike the file sink, which names it differently precisely because
    it is NOT the same measurement.
    """

    async def play(self, turn, pcm, seq=0, text="", visemes=None) -> None:
        if seq == 0:
            turn.stamp("sink_played")

    async def cancel(self, turn) -> None:
        pass


async def one_turn(reply: str) -> dict:
    orch = Orchestrator(stt=FakeStt(), llm=FakeLlm(reply), tts=FakeTts(), sink=FakeSink())
    started = time.perf_counter()
    result = await orch.run(np.zeros(16000, dtype=np.float32))
    wall_ms = (time.perf_counter() - started) * 1000
    timeline = result.turn.timeline
    return {
        "reply": reply[:40],
        "wall_ms": wall_ms,
        "headline_ms": (timeline["sink_played"] - timeline["endpoint"]) * 1000
        if "sink_played" in timeline
        else float("nan"),
        "stages": stage_breakdown(timeline),
        "error": result.error,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "bench-turn.json")
    args = ap.parse_args(argv)

    runs = [asyncio.run(one_turn(REPLIES[i % len(REPLIES)])) for i in range(args.turns)]
    failed = [r for r in runs if r["error"]]
    headlines = [r["headline_ms"] for r in runs if r["headline_ms"] == r["headline_ms"]]

    # What the fakes themselves cost. Anything above this is the
    # pipeline's own overhead, which is the only thing this can regress.
    floor_ms = STT_MS + LLM_FIRST_TOKEN_MS + TTS_FIRST_MS
    p50 = percentile(headlines, 0.50)
    p95 = percentile(headlines, 0.95)
    overhead = p50 - floor_ms

    report = {
        "turns": len(runs),
        "failed": len(failed),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "fake_provider_floor_ms": round(floor_ms, 1),
        "pipeline_overhead_ms": round(overhead, 1),
        "max_overhead_ms": MAX_OVERHEAD_MS,
        "stages_p50": {
            k: round(
                percentile(
                    [r["stages"].get(k, float("nan")) for r in runs if k in r["stages"]], 0.5
                ),
                1,
            )
            for k in {k for r in runs for k in r["stages"]}
        },
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    if failed:
        print(f"\n{len(failed)} turns FAILED: {[r['error'] for r in failed][:3]}")
        return 1
    print(f"\npipeline overhead {overhead:.1f} ms on top of {floor_ms:.0f} ms of fake providers.")
    if args.do_assert and overhead > MAX_OVERHEAD_MS:
        print(f"REGRESSION: overhead {overhead:.1f} ms exceeds {MAX_OVERHEAD_MS} ms.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
