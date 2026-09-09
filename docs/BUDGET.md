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
| TTS TTFA | 60–150 ms (Kokoro-GPU) / 300–450 ms (Qwen3-TTS) / ~470 ms (Chatterbox) | — |
| Audio out to `played` | 40–70 ms | — |
| **Sum** | **~650–950 ms** | — |

"Measured" columns are filled in as each stage lands real numbers —
never estimated or left blank once the code exists to measure them.

## Stage 0 per-component notes

- Gate G6 (Task 8) measures Kokoro's first-audio latency directly — see
  `docs/DECISIONS.md` once it runs.
- Gate G1 (Task 2) measures whether CTranslate2 reaches this GPU by
  native SASS or PTX-JIT, and records the first-vs-second-load delta.
