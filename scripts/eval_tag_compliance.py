#!/usr/bin/env python3
"""Measure the emotion-tag compliance gate against a running model.

    uv run scripts/eval_tag_compliance.py --n 60
    uv run scripts/eval_tag_compliance.py --model neiro-4b --n 200

Stage 1 gates this at >=98%. Below that the plan's fallback is the
tools-off GBNF path (grammar-constrained decoding, which llama.cpp
cannot combine with tool calling).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.config import Neiro
from neiro.evals.tag_compliance import score
from neiro.llm.ollama_native import OllamaNativeLlm
from neiro.llm.prompt import PROMPT_VERSION, system_message, user_message

# Deliberately spread across registers: flat statements, questions,
# emotional news, tool-ish requests, terse replies, and the two demo
# sentences that differ only in delivery. A prompt set of all-cheerful
# inputs would measure nothing about her range.
PROMPTS: list[tuple[str, str | None]] = [
    ("what's my battery at", None),
    ("hey", None),
    ("what's the fastest animal on land", None),
    ("i just got the build working", None),
    ("my code broke again", None),
    ("turn the volume down a bit", None),
    ("what time is it", None),
    ("i've been up since 5", "flatter and quieter than usual, tone-confidence medium"),
    ("i've been up since 5", "more energy than usual, tone-confidence medium"),
    ("yeah, I'm fine", "flatter and quieter than usual, tone-confidence high"),
    ("yeah, I'm fine", None),
    ("do you think this project is any good", None),
    ("tell me something", None),
    ("nothing", None),
    ("the tests all pass", "more energy than usual, tone-confidence high"),
    ("i can't figure out why this won't work", None),
    ("what are you", None),
    ("okay", None),
    ("guess what", None),
    ("i'm going to sleep", None),
]


async def collect(llm: OllamaNativeLlm, n: int) -> list[str]:
    replies: list[str] = []
    for i in range(n):
        text, annotation = PROMPTS[i % len(PROMPTS)]
        messages = [system_message(), user_message(text, annotation)]
        parts: list[str] = []
        try:
            async for event in llm.stream(messages):
                if "text" in event:
                    parts.append(event["text"])
        except Exception as exc:  # noqa: BLE001 — a failed turn is a data point
            print(f"  turn {i}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        replies.append("".join(parts))
        print(f"\r  {len(replies)}/{n}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return replies


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gate", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "tag-compliance.json")
    args = ap.parse_args(argv)

    cfg = Neiro()
    if args.model:
        cfg.llm.model = args.model

    replies = asyncio.run(collect(OllamaNativeLlm(cfg), args.n))
    if not replies:
        print("No replies — is the model server running?")
        return 2

    report = score(replies, gate=args.gate)
    summary = report.summary() | {"prompt_version": PROMPT_VERSION, "model": cfg.llm.model}
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    for result in report.results:
        if not result.compliant:
            print(f"\n  {result.verdict}: {result.reply[:110]!r}")

    if report.flat:
        # Not a gate failure, and said out loud anyway.
        print(
            "\nShe is COMPLIANT but FLAT — one label covers over 80% of replies. "
            "The character prompt's own words: a flat neutral on everything makes "
            "her a robot with a face bolted on. That is a prompt problem, not a "
            "parser problem, and this eval will not catch it for you."
        )
    print(f"\ngate {args.gate:.0%}: {'PASS' if report.passed else 'FAIL'} at {report.rate:.1%}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
