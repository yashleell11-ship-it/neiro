# Latency budget

The one metric, defined once, never redefined: **milliseconds from the
endpoint event (PTT release in Stages 0–1; the smart-turn decision from
Stage 2) to the browser's `played` callback for chunk seq 0.** Reported
as p50 and p95 separately — never a mean, because a semantic turn model
deliberately waits longer on unfinished-sounding speech, and an average
would make that correct behaviour look like a regression.

No two implementation options are ever compared on any other number.
Leaderboard RTFx figures are for picking a starting candidate only.

## Target (Stage 2, p50, earbuds profile)

| Row | Budget | Measured |
|---|---|---|
| Endpoint decision (Silero trigger + smart-turn) | 215–235 ms | — |
| STT finalization not hidden by prefill | 80 ms (Moonshine+prefill) / 200–350 ms (distil, no streaming) | — |
| Prefill on cached prefix | ~30 ms | — |
| TTFT (incl. ~8-token emotion tag) | 80–130 ms | — |
| Decode to first speakable clause (≤ 8 words) | 180–280 ms | — |
| TTS TTFA | 60–150 ms (Kokoro-GPU) / 300–450 ms (Qwen3-TTS) / ~470 ms (Chatterbox) | **Kokoro-CPU 334 / 365 / 769 ms** at 1 / 3 / 8 words (p50, n=12) |
| Audio out to `played` | 40–70 ms | — |
| **Sum** | **~650–950 ms** | — |

"Measured" columns are filled in as each stage lands real numbers —
never estimated or left blank once the code exists to measure them.

## Stage 0 per-component notes

- Gate G6 (2026-09-14, `scripts/bench_tts.py`, `docs/g5-g6.json`):
  Kokoro on **CPU** costs **334 ms** to first audio for a one-word
  opener, **365 ms** for three words and **769 ms** for eight (p50 over
  12 runs; p95 is within 5% of p50 at every length, so the engine is
  steady rather than occasionally slow). Warm-up after the page cache is
  hot is 4.2 s, and 64 s from cold disk — which is why `elizabeth talk`
  warms every runtime before the first turn rather than on it.

  Two consequences. **The budget's TTS row was written for Kokoro on the
  GPU and the shipped provider runs on the CPU**, where even the
  shortest opener costs more than the whole row allows; the sub-second
  target is unreachable on the CPU path at any clause length above one
  or two words. **And the "open short" rule in the prompt is worth 435
  ms**, measured — the largest single saving any prompt line has ever
  bought here, and no longer an assumption.

  What this does not yet decide: whether Kokoro on the GPU reaches the
  60–150 ms the row claims. That needs a CUDA-torch venv (the runtime
  venv deliberately holds the CPU build) and is the same run that
  settles gate G5, so both are measured together.
- Gate G1 (Task 2) measures whether CTranslate2 reaches this GPU by
  native SASS or PTX-JIT, and records the first-vs-second-load delta.
