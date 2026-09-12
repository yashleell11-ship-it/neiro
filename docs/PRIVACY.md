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

There are three tiers (`state.Tier`), and what leaves the machine
depends on which one served the turn. The rule for each provider is
declared in code (`state.Locality`) and enforced by
`Locality.allows(tier)`, which has a full truth-table test — it is an
architectural rule, not a default that could later be changed casually.

- **`local`** (the machine Yash is sitting at — the laptop, at the
  hostel): **nothing leaves.** No network at all.
- **`lan`** (home; the 3090 Ti is his own desktop, on the same router,
  over ethernet): the LLM prompt — which contains the transcript and,
  when affect is on, the `[voice: …]` annotation — and the TTS text
  cross the wire. **Raw audio crosses it only if speech-to-text is
  placed on the box** (`Locality.LAN_TIERABLE`; off unless T17a shows
  it is worth it). Both ends are his machines, but the hop is plain
  HTTP on the home LAN: llama-server's `--api-key` is authentication,
  not confidentiality. Accepted for a private wired LAN; documented so
  it isn't assumed to be more than that.
- **`tunnel`** (hostel→home via Cloudflare Access; only if T17b measures
  it fast enough): the LLM prompt and TTS text cross it, over TLS with a
  bearer token. **Raw audio never does** — STT is LAN-only by rule.

Voice-activity detection, affect detection, the audio sink, and every
tool are `Locality.LOCAL_PINNED`: they never run anywhere but the
machine in front of Yash, on any tier. There is no "remote tool" —
`capture_screen` and `read_screen_text` cannot send a screenshot over
any link because they cannot be scheduled on any tier but `local`. See
`docs/THREAT-MODEL.md` (written when Stage 4 starts), which will name
exactly what crosses the LAN versus the tunnel.

## Commands (planned, tracked here so they don't get forgotten)

- `neiro forget --since <date>` — delete turns.jsonl entries after a date.
- `neiro purge` — delete all local state, including the prosody baseline.
- `neiro pause` — stop the LLM server without deleting anything.

Until these ship, deleting `~/.local/state/neiro/` by hand does the same
thing.
