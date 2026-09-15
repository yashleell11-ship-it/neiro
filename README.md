# Elizabeth

Elizabeth is an anime character who lives on your laptop, hears *how* you sound and not
just what you said, and answers with a voice and a face driven by one shared emotional
state. Local by
default — no API keys for the voice loop itself — with two named, confirm-gated
exceptions: opening an app on Yash's own second machine, and a web search. See
`docs/PRIVACY.md`.

> **Status: early.** The voice loop runs end to end with real models. The face lands in
> Stage 1. Numbers below are measured on the machine described in *Hardware*, not
> projected — where something is unverified, it says so.

---

## What makes it different

There is no shortage of local voice assistants with anime avatars — AIRI, Open-LLM-VTuber,
Soul-of-Waifu, Amica and OpenAvatarChat all exist and several are excellent. Two things
here are not in any of them:

1. **She reads emotion from the acoustics of your speech**, not from your words, and that
   one reading reaches both the synthesiser and the face. Pitch, energy, pace and pausing
   are scored against *your own* rolling baseline — "louder and faster than you usually
   are", never "this sounds like anger in general".
2. **On Linux/Wayland she controls the desktop through a typed, tiered registry the model
   cannot escape.** The model emits an integer index or a `Literal` enum. Never a string.

The second is not paranoia. On this machine `hyprctl dispatch` is a **Lua 5.5 eval with
`os.execute` in scope** — proven by experiment during planning. The three payloads that
executed are negative tests in the suite.

## Measured, on this machine

| What | Number | Where |
|---|---|---|
| Arousal separation, Lane A prosody | **AUC 0.839** (30 speakers, per-speaker baselines) | `scripts/spike_arousal.py` |
| Valence separation | **AUC 0.511 — chance** | same |
| Emotion-tag compliance | **98–99%** (n=100, gate 98%) | `scripts/eval_tag_compliance.py` |
| Speech-emotion model (Lane B) | **CCC arousal 0.644**, speaker-independent | `training/recipes/ser_train.py` |
| STT, end of speech → transcript | **141 ms** | distil-large-v3.5 int8, GPU |
| TTS time-to-first-audio | **281 ms** (1 word) → **1088 ms** (16 words) | `docs/gate-g6-kokoro-ttfa.json` |
| Silero VAD | **0.08 ms** per 32 ms frame, CPU | `tests/test_vad.py` |
| Hyprland socket read | **0.415 ms** vs ~3 ms for `hyprctl` | `src/elizabeth/tools/hyprland.py` |

The valence number is the useful one. It says in measurement what the literature says in
prose: **how activated someone sounds is audible; whether they feel good or bad mostly is
not.** So arousal drives behaviour and valence is carried with half the confidence and its
own accessor, where nothing can inherit arousal's certainty by accident.

## The loop

```
mic ─► ring ─► VAD/endpoint ─► STT ──┐
                    │                ├─► LLM ─► <e:tag> ─► chunker ─► TTS ─► browser
                    └─► prosody ─────┘              │
                     (while you speak,              └─► face + voice, one state
                      so it costs 0 ms)
```

`Turn.user_affect` is what she *heard*. `Turn.elizabeth_state` is what she *feels*. They never
share a variable — an assistant that merges them ends up reading its own synthesised voice
back as your mood.

## Permission tiers

| Tier | Rule | Examples |
|---|---|---|
| **Green** | Runs immediately | battery, windows, what's playing |
| **Yellow** | Spoken confirmation **and** a clickable notification, in parallel — the click wins | volume, brightness, focus window |
| **Red** | Not registered as a tool at all, so the model cannot express it | shell, file writes, clipboard, `sudo` |

The confirmation nonce is computed in the registry and never sent to the model, so a reply
cannot forge one. The LLM does not participate in its own authorisation.

## Hardware

Built and measured on an ASUS TUF F16: **RTX 5070 Laptop, 8 GB (7730 MiB usable)**,
CachyOS, Hyprland 0.56.2 (**Lua** config), Wayland, PipeWire. Training happens on an
**RTX 3090 Ti, 24 GB / 48 GB RAM** at home, reached over LAN.

Every model is chosen to fit *next to* the others in 8 GB, which is a different problem
from choosing the best one.

## Running it

```bash
uv sync
source env.sh                     # ctranslate2 needs cuBLAS on the path BEFORE python starts
uv run elizabeth doctor          # checks every assumption, names the fix for each failure
uv run elizabeth fetch-models --runs local --dry-run
uv run elizabeth talk            # SPACE, speak, SPACE, hear her; SPACE while she speaks interrupts her
```

`elizabeth affect <wav>` shows the whole emotional loop for one recording without a mic, a
model or a browser: the prosody features, each z-scored against your normal, the exact
`[voice: …]` annotation the prompt would receive or why it was omitted — then the other
direction, her tag becoming face weights and a voice instruction.

## What it can't do yet

- **No face.** The VRM avatar is Stage 1. The blending, viseme timing and expression
  fallbacks are built and tested; nothing renders them.
- **No barge-in.** VAD, the speech gate and the barge-in detector exist and are tested;
  Stage 2 wires them in.
- **Slow on this laptop today.** The installed `ollama` is the CPU-only Arch build, so the
  LLM takes ~15 s. Everything either side of it is already fast. `llama-cpp` +
  `ggml-vulkan` needs a `pacman` install.
- **Emotion is off by default.** `affect.enabled` flips to true only after gate G3b passes
  on *Yash's own* recordings. Acted corpora are much easier than natural speech, and the
  published gap is enormous.

## Licence

Apache-2.0. Every model's licence is recorded in `src/elizabeth/modelspec.py`; every training
corpus's in `data/datasets.toml`, where a dataset flagged NC / ND / research-only **cannot**
claim its weights are publishable — that is a load-time error, not a release-day surprise.

The avatar is a separate download with its own terms; see `ASSETS.md`.
