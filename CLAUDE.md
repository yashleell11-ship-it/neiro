# Rules for whoever (human or agent) works on this repo

These were established by a 22-agent verification pass on 2026-09-10
that checked the original design spec against this exact machine. Full
detail: `docs/CORRECTIONS.md` (what changed and why) and
`docs/DECISIONS.md` (every gate verdict, dated).

1. **Hyprland here is the Lua config API, not hyprlang.**
   `hyprctl dispatch workspace 2` fails on this machine (0.56.2, HyDE
   layered) — the working form is `hl.dsp.focus({workspace=N})`. Almost
   every Hyprland script you'll find online (and most LLM completions)
   emit the old syntax.
   **Never string-format a value into a `hyprctl dispatch` call.** It is
   compiled as Lua 5.5 with `os.execute` in scope — proven by injection
   during the verification pass. Any tool exposed to the model must pass
   an integer index or a `Literal` enum, never a string, through
   `tools/egress.py`'s regex filter before it reaches the socket.
   Never hold the Hyprland request socket open — it evaluates
   synchronously and an unclosed connection freezes the whole desktop
   for 5 seconds. Always connect → send → read to EOF → close, with a
   timeout.

2. **DO-NOT-VENDOR**: Open-LLM-VTuber-Web (non-OSI custom licence),
   Soul-of-Waifu (GPL-3.0), LiveKit turn-detection model weights
   (proprietary), openSMILE (commercial-use forbidden in the free
   build), audeering's wav2vec2 A/D/V model (CC-BY-NC-SA).
   **DO-NOT-DEPEND** on HF-gated or non-commercial weights: Orpheus TTS
   (gated), Higgs Audio, Breeze TTS 2, OpenAudio S1, F5-TTS weights,
   IndexTTS-2.5 — all non-commercial or custom-licensed.

3. **No component's install path may require a GitHub release asset.**
   The network this project is built on blocks
   `objects.githubusercontent.com` intermittently. Everything comes from
   Hugging Face (pin the 40-char commit revision + verify sha256, never
   `main`), pacman/AUR, PyPI, or npm. Run `pacman -Si <pkg>` before
   choosing an install path — package availability has been wrong in
   research before.

4. **Thinking mode is off, and asserted, not assumed.** Every current
   Qwen3.5/3.6 checkpoint reasons by default. Set
   `--reasoning-budget 0 --reasoning-format none` on the server AND
   `chat_template_kwargs: {enable_thinking: false}` per request AND a
   runtime check that drops any turn whose reply starts with `<think` —
   the disable switch has been silently ignored on some builds.
   **System prompt bytes are byte-identical every turn**, and
   conversation history is appended, never rewritten — this is what
   makes the KV prefix cache hit; without it every turn pays a full
   prefill.

5. **`Turn.user_affect` and `Turn.neiro_state` are never assigned to
   each other.** One is what we heard in the user's voice; the other is
   what she feels. Sharing a variable is how an assistant ends up
   reading its own TTS output back as the user's mood. Below the
   configured dead-band/confidence floor, the affect annotation is
   *omitted* from the prompt entirely — never softened or guessed.

6. **The model never emits a string that reaches a command line.**
   Window/app/unit/container/file-path arguments are always an integer
   index into a list Neiro produced in the same turn, or a `Literal`
   enum member. See rule 1 for why.

7. **No raw voice audio enters git, ever.** `data/voice/**/*.wav` is
   gitignored (see `.gitignore` and `docs/PRIVACY.md`); only derived,
   non-biometric artifacts (reference transcripts, self-labels, prosody
   feature vectors, sha256 hashes) are committed.

8. **No magic numbers in modules.** Every tunable lives in `config.py`
   with a default and a comment, or it isn't tunable. **Never compare
   two implementation options on any number but the one defined metric**
   (endpoint event → browser `played` for seq 0, p50/p95, never a mean —
   see `docs/BUDGET.md`). Leaderboard/RTFx numbers are for picking a
   starting candidate, never for deciding between two things running on
   this machine.

9. **Every task ends with something Yash can see or hear**, and names
   the one concept it teaches. If a task would only produce a passing
   test with nothing observable, fold it into the next task that does.

10. **Commits pushed to Yash's GitHub carry no AI attribution** — no
    `Co-Authored-By` trailer, no "Generated with" line, no
    `@anthropic.com` author. This is a standing instruction, not a
    per-session choice.

11. **Commit every small working step, and push each sitting.** One
    commit per task or sub-step that leaves the tree working — a
    scaffold, a contract file, a passing test, a gate verdict in
    `docs/DECISIONS.md`. Never batch an evening into one commit. Push to
    the default branch: that is what the contribution graph counts, and
    the graph is part of why this repo is public.
