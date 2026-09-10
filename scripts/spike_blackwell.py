"""Gate G1 (Stage 0 Task 2): does ctranslate2 actually run int8 on this
RTX 5070 Laptop (Blackwell, compute capability 12.0), or does it silently
fall back to something slower, or crash outright?

Run via `just gate-g1`, which handles sourcing env.sh (LD_LIBRARY_PATH
must be set before this process starts — see env.sh) and the cold/warm
~/.nv/ComputeCache comparison that reveals whether the wheel has native
sm_120 SASS or reaches this GPU by one-time PTX JIT from compute_86.

Running this file directly also works, but only if you `source env.sh`
in the same shell first.

Deliberately scoped: this is a PLUMBING gate, not an accuracy gate. The
test fixture (fixtures/tone_3s.wav) is a synthesized two-tone chirp, not
real speech — proving the CUDA/int8 path executes end-to-end matters
here, not what it transcribes. Real-voice accuracy on Yash's own voice
is Gate G3a (Stage 0 Task 6), a separate, later question.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tone_3s.wav"
MODEL_ID = "distil-whisper/distil-large-v3.5-ct2"
COMPUTE_TYPE = "int8_float16"


def main() -> int:
    try:
        import ctranslate2
    except OSError as exc:
        print(
            f"FAIL: ctranslate2 failed to import: {exc}\n"
            "Most likely cause: LD_LIBRARY_PATH wasn't set before this process\n"
            "started. Run `source env.sh` in this shell first, then re-run —\n"
            "don't just `export LD_LIBRARY_PATH=...` inside Python, it's too late.",
            file=sys.stderr,
        )
        return 1

    print(f"ctranslate2 {ctranslate2.__version__}")
    supported = ctranslate2.get_supported_compute_types("cuda")
    print(f"supported cuda compute types: {sorted(supported)}")

    if COMPUTE_TYPE not in supported:
        print(
            f"\nFAIL: '{COMPUTE_TYPE}' is not in the supported set.\n"
            "This means ctranslate2 < 4.7.0 is what actually loaded — int8 on\n"
            "Blackwell (sm_120) was disabled in 4.6.2 and only re-enabled in\n"
            "4.7.0 (PR #1982, 2026-02-03). Fix: uv add 'ctranslate2>=4.7.0'.\n"
            "Do NOT fall back to compute_type='float16' to route around this —\n"
            "that masks the real problem and doubles VRAM for no reason.",
            file=sys.stderr,
        )
        return 1

    if not FIXTURE.exists():
        print(f"FAIL: test fixture missing at {FIXTURE}", file=sys.stderr)
        return 1

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(MODEL_ID, device="cuda", compute_type=COMPUTE_TYPE)
    load_s = time.perf_counter() - t0
    print(f"model load: {load_s:.2f}s")
    # Grepped by the justfile's gate-g1 recipe to compare cold vs warm runs.
    print(f"MODEL_LOAD_SECONDS={load_s:.3f}")

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        str(FIXTURE),
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=False,
        language="en",
        without_timestamps=True,
    )
    text = "".join(s.text for s in segments)
    transcribe_s = time.perf_counter() - t0
    print(f"transcribe: {transcribe_s:.3f}s")
    print(f"transcript (garbage expected — it's a synthesized tone, not speech): {text!r}")
    print(f"detected language: {info.language} (p={info.language_probability:.2f})")

    print(
        "\nPASS: int8_float16 supported, WhisperModel loaded on device='cuda',"
        "\n      and transcribe() ran end to end without error."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
