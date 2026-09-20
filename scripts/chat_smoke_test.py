#!/usr/bin/env python3
"""Talk to whatever persona checkpoint currently exists, as text.

Every training run so far has been judged by loss curves and AUC
numbers. Nobody has actually read a sentence the adapter produces.
This is the smallest thing that closes that gap: load the base model,
apply a LoRA checkpoint (not the final adapter — the in-progress
`checkpoint/` dir, so this works mid-run), render through the same
system prompt and chat template the live daemon will use, and print
what she actually says.

Not a benchmark, not gated on anything — a sanity read. Rule 9: a task
that would only produce a passing test with nothing observable doesn't
count, and eleven hours of training with nobody having read a sentence
out of it was exactly that gap.

    uv run python scripts/chat_smoke_test.py
    uv run python scripts/chat_smoke_test.py --checkpoint ../models/persona-lora/checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    from elizabeth.llm.prompt import system_message
except ModuleNotFoundError:  # pragma: no cover - the box still runs the pre-rename package
    from neiro.llm.prompt import system_message  # type: ignore[no-redef]

DEFAULT_MODEL = REPO / "models" / "qwen3.5-4b-safetensors"
DEFAULT_CHECKPOINT = REPO / "models" / "persona-lora-bilingual" / "checkpoint"

# Turns picked to exercise what v1 is actually for, which is English only
# — Hindi is paused, so a Hinglish probe here would be testing a thing we
# deliberately stopped training. Plain chat, an emotional beat that should
# draw warmth rather than a fix-it list, a British-idiom turn (the accent
# work is the whole point of the stt_english corpora), and one that invites
# a tool call so we can see whether she stays in character while acting.
SAMPLE_TURNS = [
    "hey, how's it going?",
    "I've been staring at this bug for three hours and I want to throw my laptop out the window",
    "bit knackered today, couldn't be bothered with any of it honestly",
    "can you open my browser and look up what's on at the cinema tonight?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        # Both GPUs are meant to stay pinned at 100%, so "wait for a free
        # card" means "never run". --device cpu is how the periodic quality
        # check gets to run at all: slow, but it never touches training.
        help="auto follows cuda availability; cpu forces it off a busy GPU",
    )
    ap.add_argument(
        "--tools",
        action="store_true",
        # Without this the transcript shows her SAYING she will open the
        # browser rather than calling open_app — which reads exactly like
        # a training defect and is not one: no tool was ever offered.
        help="offer the real tool schemas, so tool-calling is actually testable",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"no checkpoint at {args.checkpoint} — nothing to load yet", file=sys.stderr)
        return 1

    # Imported here, not at module level: importing torch/transformers
    # costs real seconds, and --help should stay instant.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    step = None
    step_file = args.checkpoint / "step.json"
    if step_file.exists():
        import json

        step = json.loads(step_file.read_text())["step"]

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        print("--device cuda asked for, but no CUDA device is visible", file=sys.stderr)
        return 1

    print(f"loading base model from {args.model} on {'cuda' if use_cuda else 'cpu'} ...")
    if not use_cuda:
        # bf16 on CPU is punishingly slow on most x86; fp32 is the honest
        # default here, and the 4B backbone fits in ~17 GB of RAM.
        print("cpu run: expect minutes per reply, not seconds")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    # Same NF4 load persona_train.py uses: the bf16 backbone alone is
    # 8.41 GB, more than the ~7.73 GB usable on an 8 GB card — this
    # script hit that OOM directly before matching the training recipe.
    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if use_cuda
        else None
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        quantization_config=quant_config,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map={"": 0} if use_cuda else None,
    )
    print(f"applying checkpoint {args.checkpoint}" + (f" (step {step})" if step is not None else ""))
    model = PeftModel.from_pretrained(base, str(args.checkpoint))
    model.eval()

    print(f"\n{'=' * 60}\nprompt: {system_message().get('content', '')[:80]}...\n{'=' * 60}\n")

    tool_schemas = None
    if args.tools:
        try:
            from elizabeth.tools.builtin import build_registry
        except ModuleNotFoundError:  # pragma: no cover - pre-rename box
            from neiro.tools.builtin import build_registry  # type: ignore[no-redef]

        # A confirmer that says yes: nothing is executed here, but
        # without one the registry serves GREEN schemas only and the
        # YELLOW tools this is meant to exercise never appear.
        registry = build_registry(confirm=lambda *a: True)
        tool_schemas = registry.schemas()
        print(f"offering {len(tool_schemas)} tools: {', '.join(t['function']['name'] for t in tool_schemas)}\n")

    for turn in SAMPLE_TURNS:
        messages = [system_message(), {"role": "user", "content": turn}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
            **({"tools": tool_schemas} if tool_schemas else {}),
        )
        input_ids = rendered["input_ids"].to(model.device)
        attention_mask = rendered["attention_mask"].to(model.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        reply = tokenizer.decode(generated[0][input_ids.shape[1] :], skip_special_tokens=True)
        print(f"you: {turn}")
        print(f"her: {reply.strip()}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
