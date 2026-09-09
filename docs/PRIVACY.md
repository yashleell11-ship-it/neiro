# Privacy

Neiro is a voice assistant with a live microphone. This file states
plainly what is captured, where it lives, and how to remove it. It
exists because a voice-first project that stays quiet about this is not
trustworthy, and the verification pass found the original spec never
raised it.

## What is captured

- **Live audio**, briefly, in a 30-second ring buffer in memory. Not
  written to disk during normal operation.
- **Transcripts** of what you said and Neiro's replies, written to
  `~/.local/state/neiro/turns.jsonl` for latency measurement and
  debugging. Retention: unbounded by default in v1 (no rotation yet —
  `neiro forget`/`neiro purge` are planned, not shipped).
- **Prosody calibration state** (a handful of floats — rolling mean/std
  of your pitch, energy, and speaking rate), persisted per capture
  device to `~/.local/state/neiro/baseline.json`. This is what lets
  arousal detection skip a 5-utterance warm-up every single session. It
  is *not* conversation content.
- **Recorded evaluation sets** (`data/voice/`, gitignored): WER
  utterances and, if Gate G3b passes, self-labelled emotion utterances.
  These are Yash's own voice and never leave the machine — only derived
  artifacts (transcripts, labels, feature vectors, sha256 hashes) are
  committed to the repository. See `ASSETS.md`.

## What leaves the machine

**Nothing, on the laptop-only tier.** If the 3090 Ti tier is ever built
(Stage 4, gated on a measured tunnel round trip — see
`docs/CORRECTIONS.md` #13), the LLM prompt and TTS text cross the
tunnel. Speech-to-text, voice-activity detection, affect detection, and
every tool stay laptop-pinned permanently, as an architectural rule
(`state.Locality.LAPTOP_PINNED`) — not a default that could later be
changed casually. `capture_screen` and `read_screen_text` fail closed on
the remote tier rather than silently sending a screenshot over the
tunnel. See `docs/THREAT-MODEL.md` (written when Stage 4 starts).

## Commands (planned, tracked here so they don't get forgotten)

- `neiro forget --since <date>` — delete turns.jsonl entries after a date.
- `neiro purge` — delete all local state, including the prosody baseline.
- `neiro pause` — stop the LLM server without deleting anything.

Until these ship, deleting `~/.local/state/neiro/` by hand does the same
thing.
