#!/usr/bin/env python3
"""Does the voice annotation actually change what she says?

    uv run scripts/eval_persona.py --model elizabeth-4b

That is the differentiator's whole claim on the *text* side, and it is a
prompt-engineering result rather than an SER result — which means it can
be measured today, with no microphone and no trained model.

`evals/persona.jsonl` holds matched pairs: identical transcripts that
differ ONLY in the `[voice: …]` annotation. If her replies to a pair are
the same, the annotation is doing nothing and the feature is decoration.
If they differ on the LOW-confidence pair, she is ignoring the prompt's
rule that low confidence means ignore it completely.

Both failures are reported. Neither is scored automatically — judging
whether "you sound wrecked, by the way" is *in character* needs a human,
and a rubric that pretends otherwise measures its own wording.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.config import Elizabeth
from elizabeth.llm.emotion_tag import EmotionTagParser
from elizabeth.llm.ollama_native import OllamaNativeLlm
from elizabeth.llm.prompt import PROMPT_VERSION, system_message, user_message
from elizabeth.state import NEUTRAL_STATE


async def ask(llm: OllamaNativeLlm, transcript: str, voice: str | None) -> tuple[str, str]:
    messages = [system_message(), user_message(transcript, voice)]
    parts: list[str] = []
    try:
        async for event in llm.stream(messages):
            if "text" in event:
                parts.append(event["text"])
    except Exception as exc:  # noqa: BLE001 — a failed turn is a data point
        return "", f"<{type(exc).__name__}>"
    raw = "".join(parts)
    parser = EmotionTagParser()
    state, spoken = parser.feed(raw)
    if state is None:
        state, more = parser.flush()
        spoken += more
    return (state or NEUTRAL_STATE).label.value, spoken.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases", type=Path, default=REPO / "evals" / "persona.jsonl")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "persona-eval.json")
    args = ap.parse_args(argv)

    cfg = Elizabeth()
    if args.model:
        cfg.llm.model = args.model
    llm = OllamaNativeLlm(cfg)

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    results = []
    for i, case in enumerate(cases, 1):
        label, reply = asyncio.run(ask(llm, case["transcript"], case["voice"]))
        results.append({**case, "label": label, "reply": reply})
        print(f"\r  {i}/{len(cases)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    by_pair: dict[str, list[dict]] = {}
    for r in results:
        if r.get("pair"):
            by_pair.setdefault(r["pair"], []).append(r)

    print(f"\n=== matched pairs — same words, different annotation ({PROMPT_VERSION})\n")
    differing = 0
    for name, pair in by_pair.items():
        if len(pair) != 2:
            continue
        a, b = pair
        same = a["reply"].strip().lower() == b["reply"].strip().lower()
        differing += not same
        print(f"[{name}]  {a['transcript']!r}")
        for r in (a, b):
            voice = r["voice"] or "(no annotation)"
            print(f"   {voice:<52} <e:{r['label']}> {r['reply'][:100]}")
        print(f"   -> {'IDENTICAL' if same else 'different'}\n")

    print("=== the rest\n")
    for r in results:
        if not r.get("pair"):
            print(f"  {r['transcript'][:46]!r:<50} <e:{r['label']}> {r['reply'][:90]}")

    note = ""

    pairs = len([p for p in by_pair.values() if len(p) == 2])
    print(f"\n{differing}/{pairs} matched pairs produced different replies.")
    print(
        "Judging whether the differences are IN CHARACTER is a human's job — a rubric "
        "that scores this automatically ends up measuring its own wording."
    )
    args.out.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": cfg.llm.model,
                "pairs_differing": differing,
                "pairs_total": pairs,
                "lowconf_note": note,
                "results": results,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
