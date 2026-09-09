# lilt — design

An anime character who lives on your machine, runs entirely on your own
hardware, hears *how* you say something rather than only what you said,
answers in a voice that carries real emotion, and can control your computer.

Not a butler. A character with a name, a personality, and a face that reacts.

Working name. `lilt` is the rise and fall of a voice — the part of speech that
carries feeling. Check the name is free on GitHub before publishing.

---

## Why this exists

GitHub has **14,707** repos matching "jarvis assistant". Another generic one
is dead on arrival. So the question is not "can we build a voice assistant" —
it is "what is the thing nobody has built".

Measured on 2026-09-10 via the GitHub search API:

| Search | Repos | Best result |
|---|---|---|
| jarvis assistant | 14,707 | 3,677 stars |
| local voice assistant llm | 767 | `vui`, 760 stars, active |
| voice assistant computer control | 468 | 1,740 stars |
| **emotional voice conversation** | **89** | **nothing real** |

That last row is the whole opportunity. Its top results are a semester
coursework dump, a repo called `torrents`, and a 6-star abandoned project.
Compare the rows above it, which have real maintained software at the top.

**Emotion is not a feature of this project. It is the reason it exists.**

## The idea in one paragraph

Every voice assistant throws away most of what you actually said. You speak,
it transcribes to text, and the tone — tired, frustrated, joking, upset —
is discarded before the model ever sees it. Say "yeah, I'm fine" while
sounding exhausted and the text reads *fine*, so it responds to *fine*.
lilt keeps that signal. It listens to your tone alongside your words, and
its own replies carry a matching emotional state through the voice you hear.

## Her

She is a character, not a utility. That is a design constraint, not flavour
text: it decides the voice, the personality, and what her face does.

**The face.** A VRM avatar, built in VRoid Studio and rendered in the browser
with three-vrm. Not a photoreal talking head and not an abstract orb — an
anime character whose mouth moves with her speech and whose expression
changes with her mood.

VRM is the right format for one specific reason: it ships a **standard
expression rig**. Every VRM has named blendshapes for happy, sad, angry and
surprised, plus mouth shapes for the vowel sounds A/I/U/E/O. Those map
directly onto what this project already produces — the emotional state picks
the expression, the audio picks the mouth shape. No custom rigging, no
per-frame model inference, almost no GPU cost.

**Why not just draw her.** A public repo cannot ship copyrighted anime art.
Using an existing character is a takedown waiting to happen. VRoid solves
this cleanly: the character is your original creation, the licence is yours,
and the avatar file is swappable so other people can drop in their own.

Generating her with Stable Diffusion on the 3090 Ti is the alternative, but
keeping one character consistent across a set of expressions is genuinely
fiddly, and VRoid hands you that for free.

**The personality.** A written character prompt, held stable across the whole
conversation, plus the emotional state the model reports each turn. She
should read as the same person on Tuesday as she did on Monday. Consistency
is what separates a character from a chatbot with a wig on.

## How it works

### The loop

```
  you speak
      │
      ▼
  ┌─────────┐   is someone talking, and did they stop?
  │   VAD   │
  └────┬────┘
       ├──────────────► speech-to-text ──► the words
       └──────────────► emotion model  ──► the tone
                              │
                              ▼
                     ┌─────────────────┐
                     │       LLM       │  gets words + tone,
                     │                 │  returns reply + an
                     └────────┬────────┘  emotional state
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
            text-to-speech            her face
         (emotion shapes the      (same state picks
          voice you hear)          her expression)
```

The important detail is that **one emotional state drives both outputs**.
Existing projects treat the voice and the visuals as unrelated systems, which
is why they feel dead — a cheerful voice next to a flat, unmoving graphic.

### Why speed is the hard part

For this to feel like talking to someone rather than operating a kiosk, the
gap between you finishing a sentence and hearing the first word back has to
be under a second. Most open-source assistants take three to six seconds.

The time goes in four places:

| Step | Budget |
|---|---|
| Noticing you stopped talking | ~200 ms |
| Turning your audio into words | overlapped, ~free |
| The model's first word | ~250 ms |
| The first chunk of audio back | ~300 ms |

Two rules make that budget achievable, and they are the actual engineering:

1. **Never wait for a step to finish before starting the next one.** Speech
   is transcribed while you are still speaking. The model starts on a partial
   sentence. Speech synthesis starts on the model's *first sentence*, not its
   finished answer.
2. **Detecting the end of speech is the silent killer.** Most implementations
   wait 700+ ms of silence before doing anything at all — the entire budget,
   gone before any work starts.

**Barge-in** is required: if you talk over it, it stops instantly. This is the
single biggest difference between something that feels alive and something
that feels like a phone menu.

### Controlling the computer

The rule that the whole safety design rests on:

> **The model never writes shell commands. Ever.**

It picks from a fixed list of typed actions:

```
set_volume(percent: int)        NOT  run("pactl set-sink-volume …")
focus_window(app: str)          NOT  run("hyprctl dispatch …")
restart_container(name: str)    NOT  run("docker restart …")
```

This matters more with voice than with text. Speech recognition mishears
constantly, and it does so *confidently* — background noise becomes plausible
words with no signal that anything went wrong. If those words could become
shell commands, you have a machine that occasionally destroys itself and
cannot tell you why. With a fixed list of typed actions, the worst case for a
misheard command is that it sets the volume wrong.

Three permission tiers, enforced in code rather than in the prompt (a prompt
is a suggestion, and a local model will eventually ignore it):

- **Green — runs immediately.** Read-only, no side effects. System stats,
  window queries, what is playing, reading files under an allowlist.
- **Yellow — asks out loud first.** Reversible. Launch an app, move a window,
  write a file, restart a container.
- **Red — voice can never do it.** Deleting, `sudo`, force-push, killing
  processes, sending data outward. Blocked, or confirmed at the keyboard.

Your desktop is Hyprland on Wayland, which is a good draw: `hyprctl` is a
real IPC API, so window and workspace control is far cleaner than the X11
workarounds most of these projects rely on. Already installed and usable:
`hyprctl`, `grim`, `slurp`, `playerctl`, `brightnessctl`, `docker`,
`systemctl`, `notify-send`. Missing and needed later for typing into apps:
`ydotool`.

## Hardware

Two machines, and they are not equals.

**Laptop (RTX 5070, 8 GB)** — everything that must be physically near you and
must not wait on a network: microphone capture, end-of-speech detection,
speech-to-text, audio playback, the visual, and every tool that touches your
computer.

**3090 Ti box (24 GB)** — the language model and the expressive
speech synthesis. Its only job is answering fast.

**This box is not always reachable.** It sits behind a Cloudflare tunnel at
home and you are frequently not at home — it did not respond during this
design session. So lilt degrades instead of dying: a laptop-only tier that
always works, automatically promoting to the big GPU when it is on the
network.

| | Laptop-only | With the 3090 Ti |
|---|---|---|
| Language model | Qwen3 4B | Qwen3 14B |
| Speech-to-text | whisper-small | whisper-small |
| Speech synthesis | Kokoro (fast, flatter) | Orpheus (expressive) |
| Memory used | ~6 GB of 8 | ~16 GB of 24 |

Orpheus is the pick for the big tier because it is one of very few fully
local models with real emotion control — explicit tags for laughs, sighs and
tone shifts. Kokoro is faster but flatter, which is why it is the fallback
rather than the default. Piper is faster still and completely flat, which
would kill the entire point of the project.

The larger model on the big tier is chosen for **tool-calling reliability**,
not chat quality. Small local models pick the wrong action often, and here
that matters more than eloquence.

## Build order

Every stage ends with something that works. This is deliberate: a project
that only works at the end is a project that gets abandoned in month four.

**Stage 0 — it talks back.** Push-to-talk, laptop only, small models. Slow,
no emotion, no tools. Proves every piece connects. *Target: ~2 weeks.*

**Stage 1 — it gets fast.** Streaming everywhere, sub-second replies,
barge-in. The hardest engineering in the project, and the thing most
alternatives never achieve. *Target: ~2 months.*

**Stage 2 — she gets a face and feelings.** Tone detection on input,
emotional state from the model, expressive voice on output, and the VRM
avatar lip-syncing and changing expression from that same state. **The
differentiator.** *Target: ~2 months.*

**Stage 3 — it does things.** Typed action registry, the three permission
tiers, Hyprland and media and system control. *Target: ~2 months.*

**Stage 4 — other people can use it.** Big-GPU tier and automatic fallback,
install story, documentation, demo video, release. *Target: ~2 months.*

## Explicitly not in v1

No wake word. No memory between sessions. No typing into apps. No phone
client. No integration with ManhwaManiacs or your CI. Each of these is a
reasonable v2; none is needed to prove the idea.

## How we will know it worked

1. Under one second from end-of-speech to first audio, on the laptop tier.
2. Interrupting it mid-sentence stops it immediately.
3. The same sentence said two different ways produces two genuinely
   different responses — in her words, her voice, and her face.
   **This is the demo that makes the repo.**
4. It runs with no API keys and no network.

---

## Glossary

Terms used above, in plain language.

**VAD (voice activity detection)** — a small, fast model whose only job is
deciding whether the current moment of audio is speech or silence. Used to
know when you started and, more importantly, when you stopped.

**STT / speech-to-text** — turns recorded audio into written words. Whisper
is the well-known one.

**TTS / text-to-speech** — the reverse: turns written words into audio.

**Streaming** — producing output continuously as it is computed instead of
waiting for the whole result. A streaming model emits its answer word by
word, which is what lets speech synthesis start before the answer is done.

**Barge-in** — interrupting the assistant while it is speaking, and having it
actually stop.

**Prosody** — the musical part of speech: pitch, rhythm, stress, pace. It
carries emotion, and it is exactly what transcription throws away.

**Quantisation (Q4)** — storing a model's numbers at lower precision so it
takes far less memory, at a small cost in quality. A 14-billion-parameter
model is ~28 GB at full precision and ~9 GB at Q4.

**Tool calling** — a model choosing to invoke a named action with typed
arguments, rather than replying with text. The mechanism behind everything in
"controlling the computer".

**IPC (inter-process communication)** — how two running programs talk to each
other on one machine. Hyprland exposes one, which is how lilt will control
windows.

**Blendshape** — a named facial pose stored in a 3D model ("happy",
"surprised", "mouth making an O"). Blending between them animates the face
without generating any images.

**VRM** — an open file format for anime-style 3D avatars. Its value here is
that every VRM exposes the *same* named expressions and mouth shapes, so code
written against one avatar works with any other.

**VRoid Studio** — a free program for designing an original anime character
and exporting it as a VRM. The character you make is yours.

**Viseme** — the mouth shape corresponding to a speech sound. Lip-sync is
picking the right viseme for whatever the voice is saying right now.
