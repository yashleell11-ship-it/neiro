#!/usr/bin/env python3
"""Gate G2 — does everything fit in 7730 MiB at the same time?

    source env.sh && uv run scripts/spike_vram.py
    uv run scripts/spike_vram.py --skip-llm     # when llama-server is not installed

Loads each component in the order a real session does and reports VRAM
after every step, because the number that matters is not any one model's
footprint — it is the total with all of them resident *and* a browser
tab rendering a VRM.

**Why this is a gate and not a note.** VRAM does not swap. When it runs
out, whichever process allocated last dies, and that is usually not the
one that was greedy. The plan's no-go ladder is pre-decided so the
decision is not made under pressure:

    1. STT to Moonshine on CPU      (0 VRAM, ~269 ms added)
    2. drop MTP / speculative decode
    3. llama-server -c 4096 instead of 8192

GO is under 7.0 GB of the 7730 MiB usable, leaving headroom for the
compositor and whatever else is open. There is also a "not Neiro" row:
Steam and Rocket League share this card, which is what `neiro pause` is
for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

USABLE_MIB = 7730  # measured, not the 8151 the card advertises
GO_THRESHOLD_GB = 7.0


def vram_mib() -> tuple[int, int]:
    """`(used, total)` in MiB across the whole card, not just this process.

    Whole-card on purpose: the compositor, the browser and any game are
    all competing for the same memory, and a per-process number would
    say everything is fine right up until something dies.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        used, total = (int(x.strip()) for x in out.split(",")[:2])
        return used, total
    except Exception:  # noqa: BLE001 — no GPU is a legitimate answer
        return 0, 0


def processes() -> list[tuple[str, int]]:
    """Per-process VRAM, so a surprise has a name."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        rows = []
        for line in out.splitlines():
            if "," in line:
                name, mib = line.rsplit(",", 1)
                rows.append((Path(name.strip()).name, int(mib.strip())))
        return rows
    except Exception:  # noqa: BLE001
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-llm", action="store_true", help="llama-server not installed yet")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "gate-g2-vram.json")
    args = ap.parse_args(argv)

    baseline, total = vram_mib()
    if total == 0:
        print("No NVIDIA GPU visible — nothing to measure.")
        return 2

    steps: list[dict] = []

    def record(label: str, note: str = "") -> None:
        used, _ = vram_mib()
        steps.append({"step": label, "used_mib": used, "delta_mib": used - baseline, "note": note})
        print(f"  {label:<34} {used:>5} MiB  (+{used - baseline:>4} since start)  {note}")

    print(
        f"card: {total} MiB total, {USABLE_MIB} MiB usable, {baseline} MiB in use before anything\n"
    )
    record("baseline", "compositor and whatever is already open")

    # 1. STT — the first thing a turn needs, and the first thing the
    #    no-go ladder moves to CPU.
    try:
        from neiro.stt.faster_whisper import FasterWhisperStt

        stt = FasterWhisperStt()
        stt.warm()
        record("+ STT (distil int8)", "ladder step 1 moves this to CPU")
    except Exception as exc:  # noqa: BLE001 — a gate reports, it does not crash
        print(f"  STT failed: {type(exc).__name__}: {str(exc)[:80]}")
        print("  (if this is libcublas, you did not `source env.sh`)")
        stt = None

    # 2. TTS. Kokoro is CPU by design, so this row should read ~0 —
    #    proving it is the point, not assuming it.
    if not args.skip_tts:
        try:
            from neiro.tts.kokoro import KokoroTts

            tts = KokoroTts()
            tts.warm()
            record("+ TTS (Kokoro, CPU)", "expected ~0; CPU torch by design")
        except Exception as exc:  # noqa: BLE001
            print(f"  TTS failed: {type(exc).__name__}: {str(exc)[:80]}")

    # 3. VAD and endpointing — both CPU ONNX, both expected ~0.
    try:
        from neiro.audio.endpoint import SmartTurnEndpointer
        from neiro.audio.vad import SileroVad

        SileroVad().warm()
        SmartTurnEndpointer().warm()
        record("+ VAD & smart-turn (CPU ONNX)", "expected ~0")
    except Exception as exc:  # noqa: BLE001
        print(f"  VAD/endpoint failed: {type(exc).__name__}")

    # 4. The LLM is the big one and it is a separate process, which is
    #    why this reads the whole card rather than this process.
    if not args.skip_llm:
        used, _ = vram_mib()
        if used - baseline < 1000:
            print("  (no llama-server running — start it first, or pass --skip-llm)")
        record("+ LLM (if running)", "the 3.6-4.0 GB row in the ledger")

    print()
    for name, mib in processes():
        print(f"  process {name:<28} {mib:>5} MiB")

    peak = max((s["used_mib"] for s in steps), default=0)
    peak_gb = peak / 1024
    verdict = "GO" if peak_gb < GO_THRESHOLD_GB else "NO-GO"
    report = {
        "gate": "G2",
        "usable_mib": USABLE_MIB,
        "peak_used_mib": peak,
        "peak_used_gb": round(peak_gb, 2),
        "threshold_gb": GO_THRESHOLD_GB,
        "verdict": verdict,
        "steps": steps,
        "processes": [{"name": n, "mib": m} for n, m in processes()],
        "missing": ["llama-server" if args.skip_llm else None, "browser VRM tab"],
        "ladder": [
            "STT to Moonshine on CPU (0 VRAM, ~269 ms added)",
            "drop MTP / speculative decode",
            "llama-server -c 4096 instead of 8192",
        ],
    }
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\npeak {peak} MiB ({peak_gb:.2f} GB) of {USABLE_MIB} MiB usable — {verdict}")
    if args.skip_llm:
        print("INCOMPLETE: the LLM is the largest row and was not measured.")
    print("Also unmeasured: a Chromium tab rendering the VRM (0.25-0.4 GB expected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
