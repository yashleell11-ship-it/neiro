# Privacy

Elizabeth is a voice assistant with a live microphone. This file states
plainly what is captured, where it lives, and how to remove it. It
exists because a voice-first project that stays quiet about this is not
trustworthy, and the verification pass found the original spec never
raised it.

## What is captured

- **Live audio**, briefly, in a 30-second ring buffer in memory. Not
  written to disk during normal operation.
- **Transcripts** of what you said and Elizabeth's replies, written to
  `~/.local/state/elizabeth/turns.jsonl` for latency measurement and
  debugging. Retention: unbounded by default in v1 (no rotation yet —
  `elizabeth forget`/`elizabeth purge` are planned, not shipped).
- **Prosody calibration state** (a handful of floats — rolling mean/std
  of your pitch, energy, and speaking rate), persisted per capture
  device to `~/.local/state/elizabeth/baseline.json`. This is what lets
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
  hostel): **nothing leaves, with two named exceptions** — see "Tool
  exceptions" below. Neither is a provider tier and neither is silent.
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

## Tool exceptions (2026-09-15, docs/DECISIONS.md)

Two tools are exceptions to "`local` tier ⇒ nothing leaves," on purpose,
and both are YELLOW: spoken confirmation *and* a clickable notification
are required before either runs, and both are audited like every other
tool (`~/.local/state/elizabeth/audit.jsonl`).

- **`open_app(target="pc", ...)`** reaches the 3090 Ti box over
  Tailscale to launch an app there. Still just Yash's own two machines
  — the same pair `lan`/`tunnel` already cover for the LLM provider —
  never an arbitrary host.
- **`web_search`** reaches the open internet (DuckDuckGo, no API key).
  This is the one tool in the whole registry that isn't Yash's own
  hardware on the other end. It exists because RED tier already draws
  the line at "sending anything off-machine the user didn't name" — a
  search query he asked for this turn is the opposite of that.

**Not yet done, and worth doing before either ships live:** neither
tool's registration currently checks which provider tier is serving the
turn. Routing a request through the box or a tunnel and then also
reaching the box again, or the internet, compounds exactly the
confidentiality question this file exists to be honest about — the
registry only gates on GREEN/YELLOW/RED (tools/tiers.py) today, not on
`state.Tier` (LOCAL/LAN/TUNNEL). Flagged here rather than left
implicit; see docs/DECISIONS.md.

Voice-activity detection, affect detection, the audio sink, and every
tool are `Locality.LOCAL_PINNED`: they never run anywhere but the
machine in front of Yash, on any tier. There is no "remote tool" —
`capture_screen` and `read_screen_text` cannot send a screenshot over
any link because they cannot be scheduled on any tier but `local`. See
`docs/THREAT-MODEL.md` (written when Stage 4 starts), which will name
exactly what crosses the LAN versus the tunnel.

## Commands (planned, tracked here so they don't get forgotten)

- `elizabeth forget --since <date>` — delete turns.jsonl entries after a date.
- `elizabeth purge` — delete all local state, including the prosody baseline.
- `elizabeth pause` — stop the LLM server without deleting anything.

Until these ship, deleting `~/.local/state/elizabeth/` by hand does the same
thing.
