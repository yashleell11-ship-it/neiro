#!/usr/bin/env python3
"""Gates G5 and G6: how long before she makes a sound, and what it costs.

**The only number that matters here is time-to-first-audio (TTFA)**, and
specifically TTFA for the *first clause of a reply* — not the time to
synthesise a whole sentence, and never a real-time factor. The user
hears nothing until the first chunk exists, so TTFA is the part of the
budget a TTS engine spends; everything after it is hidden behind
playback as long as the engine keeps up.

**Which is why the first clause is measured at three lengths.** TTFA
scales with how much text the engine is handed, so "Kokoro is 90 ms" is
meaningless without saying 90 ms *for what*. The prompt tells her to
open with a short clause precisely because of this, and that rule is
only worth its complexity if the numbers here justify it. One, three and
eight words: the shortest plausible opener, a natural one, and the
prompt's stated ceiling.

**Gate G6** (Kokoro, the Stage 0 voice) asks what the current voice
actually costs, so `docs/BUDGET.md` stops carrying a guess.
**Gate G5** (Chatterbox, Qwen3-TTS) asks whether an *expressive* voice
fits on this card at all: GO is TTFA under 500 ms with the LLM resident,
and peak VRAM that leaves room for everything else in the ledger.

Engines are skipped, loudly and by name, when their package or weights
are missing — the CUDA-torch engines deliberately are not runtime
dependencies of this venv (see the module docstrings in `tts/`), so
"chatterbox: skipped" is the expected result here until G5 is run in a
venv that has it.

    uv run scripts/bench_tts.py                     # whatever is installed
    uv run scripts/bench_tts.py --engine kokoro -n 20
    uv run scripts/bench_tts.py --json docs/g5-g6.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.config import Neiro
from neiro.metrics import percentile
from neiro.state import EmotionLabel, NeiroState

SAMPLERATE = 24000

# The first clause of a reply, at the three lengths that bracket the
# prompt's "open short" rule. Real sentences, not lorem: phoneme count
# and punctuation both move TTFA.
# Keyed by their own word count, computed rather than asserted: a
# hand-written key that disagreed with the text silently dropped the
# whole eight-word row from the first run of this bench.
_CLAUSE_TEXTS = (
    "Sure.",
    "It is noon.",
    "Ninety six percent, still charging, about an hour.",
)
CLAUSES = {len(t.split()): t for t in _CLAUSE_TEXTS}
assert len(CLAUSES) == len(_CLAUSE_TEXTS), "two clauses have the same word count"

# The state passed to every engine that can use one. Deliberately not
# neutral: an expressive engine doing nothing is the failure G5 exists
# to catch, and a flat state would hide it.
STATE = NeiroState(label=EmotionLabel.HAPPY, intensity=0.7, valence=0.6, arousal=0.5)


def vram_mb() -> float | None:
    """Peak VRAM this process has allocated, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:  # noqa: BLE001 — a bench must never die on its own instrumentation
        return None


def reset_vram() -> str:
    """Returns what happened, so a silent no-op is visible in the log."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "no cuda"
        torch.cuda.reset_peak_memory_stats()
        return "reset"
    except Exception as exc:  # noqa: BLE001 — instrumentation never fails a bench
        return f"unavailable ({type(exc).__name__}: {exc})"


@dataclass
class Run:
    """One synthesis of one clause."""

    words: int
    ttfa_ms: float
    total_ms: float
    audio_s: float
    chunks: int


@dataclass
class EngineResult:
    name: str
    available: bool
    reason: str = ""
    warm_ms: float = 0.0
    peak_vram_mb: float | None = None
    runs: list[Run] = field(default_factory=list)

    def by_clause(self) -> dict:
        out = {}
        for words in sorted({r.words for r in self.runs}):
            ttfa = [r.ttfa_ms for r in self.runs if r.words == words]
            audio = [r.audio_s for r in self.runs if r.words == words]
            out[str(words)] = {
                "n": len(ttfa),
                # p50/p95, nearest-rank. Never a mean: a bench that
                # averages hides exactly the occasional slow first chunk
                # that a listener notices.
                "ttfa_p50_ms": round(percentile(ttfa, 50), 1),
                "ttfa_p95_ms": round(percentile(ttfa, 95), 1),
                "audio_s": round(statistics.median(audio), 2),
            }
        return out


async def time_one(engine, text: str, state: NeiroState | None) -> Run:
    """TTFA is measured to the first chunk that carries samples.

    Some engines yield an empty leading chunk; counting that as "first
    audio" would report a number the listener never experiences.
    """
    started = time.perf_counter()
    first: float | None = None
    samples = 0
    chunks = 0
    async for pcm, _visemes in engine.synth(text, state):
        size = int(np.size(pcm))
        if size == 0:
            continue
        chunks += 1
        samples += size
        if first is None:
            first = time.perf_counter() - started
    total = time.perf_counter() - started
    if first is None:
        raise RuntimeError("engine produced no audio at all")
    return Run(
        words=len(text.split()),
        ttfa_ms=first * 1000,
        total_ms=total * 1000,
        audio_s=samples / SAMPLERATE,
        chunks=chunks,
    )


def build(name: str, cfg: Neiro, device: str):
    """Import lazily and per engine: importing chatterbox or qwen3tts
    pulls a CUDA torch path that must not be loaded just to bench Kokoro.
    """
    if name == "kokoro":
        from neiro.tts.kokoro import KokoroTts

        return KokoroTts(cfg), None
    if name == "chatterbox":
        from neiro.tts.chatterbox import ChatterboxTts

        return ChatterboxTts(cfg=cfg, device=device), STATE
    if name == "qwen3tts":
        from neiro.tts.qwen3tts import Qwen3Tts

        return Qwen3Tts(cfg=cfg, device=device), STATE
    raise SystemExit(f"unknown engine {name!r}")


async def bench(name: str, cfg: Neiro, repeats: int, device: str) -> EngineResult:
    result = EngineResult(name=name, available=False)
    try:
        engine, state = build(name, cfg, device)
    except ImportError as exc:
        result.reason = f"package not installed ({exc})"
        return result

    reset_vram()
    try:
        started = time.perf_counter()
        warm = getattr(engine, "warm", None)
        if warm is not None:
            warm()
        else:  # pragma: no cover — every engine here has warm()
            await time_one(engine, "Hi.", state)
        result.warm_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001 — a missing engine is a result, not a crash
        result.reason = f"{type(exc).__name__}: {exc}"
        return result

    result.available = True
    for _ in range(repeats):
        for text in CLAUSES.values():
            result.runs.append(await time_one(engine, text, state))
    result.peak_vram_mb = vram_mb()
    return result


def table(results: list[EngineResult]) -> str:
    lines = [
        f"{'engine':<12} {'warm':>8} {'VRAM':>9}  "
        + "  ".join(f"{w}w p50/p95 ms".rjust(18) for w in sorted(CLAUSES))
    ]
    for r in results:
        if not r.available:
            lines.append(f"{r.name:<12} skipped: {r.reason}")
            continue
        by = r.by_clause()
        vram = f"{r.peak_vram_mb:.0f} MB" if r.peak_vram_mb else "cpu"
        cells = []
        for w in sorted(CLAUSES):
            c = by.get(str(w))
            cells.append(
                f"{c['ttfa_p50_ms']:.0f} / {c['ttfa_p95_ms']:.0f}".rjust(18) if c else "-".rjust(18)
            )
        lines.append(f"{r.name:<12} {r.warm_ms:>7.0f}ms {vram:>9}  " + "  ".join(cells))
    return "\n".join(lines)


def verdicts(results: list[EngineResult], budget_ms: float) -> list[str]:
    """G5 is a gate with a number, so say the number and the verdict.

    Judged on the longest clause, not the shortest: the ceiling the
    prompt allows is what has to fit the budget, and picking an engine on
    its best case is how a budget is missed in use.
    """
    longest = str(max(CLAUSES))
    out = []
    for r in results:
        if not r.available:
            continue
        cell = r.by_clause().get(longest)
        if cell is None:  # pragma: no cover
            continue
        p50 = cell["ttfa_p50_ms"]
        verdict = "GO" if p50 < budget_ms else "NO-GO"
        if r.name == "kokoro":
            out.append(f"G6 Kokoro: TTFA p50 {p50:.0f} ms at {longest} words (budget line only)")
        else:
            out.append(
                f"G5 {r.name}: TTFA p50 {p50:.0f} ms at {longest} words vs {budget_ms:.0f} ms "
                f"→ {verdict}" + (f", peak {r.peak_vram_mb:.0f} MB VRAM" if r.peak_vram_mb else "")
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--engine",
        action="append",
        choices=["kokoro", "chatterbox", "qwen3tts"],
        help="repeatable; default is all three, skipping what is not installed",
    )
    ap.add_argument("-n", "--repeats", type=int, default=10, help="runs per clause length")
    ap.add_argument("--device", default="cuda", help="for the CUDA engines")
    ap.add_argument(
        "--budget-ms",
        type=float,
        default=500.0,
        help="gate G5's TTFA line; see docs/BUDGET.md",
    )
    ap.add_argument("--json", type=Path, help="write the full result here")
    args = ap.parse_args(argv)

    cfg = Neiro()
    engines = args.engine or ["kokoro", "chatterbox", "qwen3tts"]
    results = [asyncio.run(bench(name, cfg, args.repeats, args.device)) for name in engines]

    print(table(results))
    print()
    for line in verdicts(results, args.budget_ms):
        print(line)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "clauses": CLAUSES,
                    "repeats": args.repeats,
                    "budget_ms": args.budget_ms,
                    "engines": [
                        {
                            "name": r.name,
                            "available": r.available,
                            "reason": r.reason,
                            "warm_ms": round(r.warm_ms, 1),
                            "peak_vram_mb": r.peak_vram_mb,
                            "by_clause": r.by_clause(),
                        }
                        for r in results
                    ],
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.json}")

    return 0 if any(r.available for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
