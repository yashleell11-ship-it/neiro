# Corrections to the original design spec

The design spec (`docs/superpowers/specs/2026-09-10-lilt-design.md`) was
written before a 22-agent verification pass checked its claims against
the live model ecosystem and this exact machine. Every load-bearing
claim in it was refuted or made conditional. This file is the permanent
record of what changed and why; the spec itself is kept as the record
of original intent, not edited in place.

Full detail lives in the build plan
(`/home/yash/.claude/plans/now-plan-everyhting-u-fuzzy-kernighan.md`
as of 2026-09-10) under "What the verification changed". Summary:

1. **The "89 repos" market-sizing claim is cut.** It was a search-phrasing
   artifact — AIRI (~45k★, MIT, VRM, Linux), Open-LLM-VTuber (13.7k),
   Soul-of-Waifu, Amica, and OpenAvatarChat all exist and are closer
   competitors than the spec acknowledged. The real, checkable claim:
   no shipped project derives emotion from the *acoustics* of the
   user's speech and lets one state reach both the synthesiser and the
   face; on Linux/Wayland, none controls the desktop through a typed,
   tiered registry the model cannot escape.

2. **Barge-in and sub-second latency are not rare** — GLaDOS, vui,
   Vocalis, and Open-LLM-VTuber all ship both. Kept as requirements;
   the spec's claims of rarity are dropped.

3. **Kokoro has zero emotion control**, not "flatter" — no vector, no
   tags, no instruct field. It is Stage 0 plumbing only. The expressive
   voice is Qwen3-TTS-1.7B (Apache-2.0, free-form `instruct` string) if
   Gate G5 passes, with Chatterbox-Turbo (MIT) as the fallback. Orpheus
   is dropped: 18 months stale, HF-gated (breaks "no API keys"), and
   its "emotion control" is eight sound-effect tags, not an emotional
   register.

4. **whisper-small is replaced by distil-large-v3.5** on the Indian-accent
   leaderboard column (3.60 WER vs whisper-large-v3's 3.95; whisper-small
   isn't listed at all), MIT, ~0.8 GB int8.

5. **"Qwen3 14B" doesn't exist** in the current model generation. Laptop:
   Qwen3.5-4B. Box (if it earns a place): Qwen3.6-35B-A3B.

6. **SER (speech emotion recognition) is far weaker than the spec assumed.**
   Best system on earth: macro-F1 0.43 on natural conversational speech
   (8 classes); 0.65–0.92 on *acted*, deliberately-contrasted delivery.
   No published number exists for Indian-accented English. Consequence:
   v1 ships Lane A only (prosody z-scores vs a personal baseline,
   librosa), default OFF until Gate G3b measures it on Yash's own
   voice, and the demo (deliberate contrast) is claimed — everyday
   subtle mood tracking is explicitly not.

7. **VRM 1.0 mouth presets are `aa ih ou ee oh`**, not the spec's
   A/I/U/E/O (that's VRM 0.x). VRoid Studio has no Linux build (Steam
   Proton, ProtonDB Gold). The exported VRM's base mesh is pixiv
   content, explicitly not CC0 — the repo ships a CC0 avatar as default;
   an original VRoid avatar (if made) ships separately with its own
   `ASSETS.md` entry.

8. **The spec's central safety claim is false on this machine.**
   `hyprctl dispatch X` is a Lua 5.5 eval with `os.execute` in scope —
   proven by injection. See CLAUDE.md rule 1 and rule 6.

9. **Docker is not usable locally** (daemon disabled, user not in the
   group) — podman works. `wtype` (MIT, no daemon) is preferred over
   `ydotool` for any future typing tool. File *listing* is Green tier;
   file *content* reads are Yellow with deny-globs (`~/.ssh`, `.env`,
   `*.pem` all sit under plausible allowlist roots).

10. **Default audio is Bluetooth earbuds** (A2DP — no mic until an HFP
    switch), not built-in mic/speakers as the spec assumed. Built-ins
    are muted. Two named device profiles (earbuds, speakers) from
    Stage 0 Task 0; earbuds are primary/demo, speakers are
    best-effort with echo cancellation.

11. **The real VRAM ceiling is ~7730 MiB**, not 8151 — and the
    compositor uses ~18 MiB, not the 0.5–1.5 GB the spec guessed. The
    GPU MUX is discrete-only, so there's no iGPU offload trick
    available for the browser's WebGL context.

12. **The 1000 ms latency budget was missing three rows**: STT
    finalization after endpointing, decode-to-first-*clause* (no TTS
    accepts a single token), and the audio-output path. See
    `docs/BUDGET.md`.

13. **The 3090 Ti tier's viability is unmeasured**, not assumed:
    `cloudflared` isn't installed, the box sits behind Cloudflare
    Access (a credential dependency), and every tool call would cross
    the tunnel because tools are laptop-pinned. Stage 0 Task 17
    measures the round trip before Stage 4 commits to building it.

14. **The prosody calibration baseline is memory**, and the spec's "no
    memory between sessions" rule was being misapplied to it. Baseline
    state persists per device (with drift detection); conversation
    history still does not.

15. **Blackwell (sm_120) support is real but conditional**, not a given:
    torch's default PyPI wheel and onnxruntime-gpu ≥1.27 have native
    sm_120 kernels; CTranslate2 does not (PTX-JIT from sm_86) and needs
    ≥4.7.0 for int8 to work at all. `neiro doctor` asserts the CUDA
    provider is actually *used*, never just listed as available.
