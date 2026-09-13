# Architecture

How Neiro is put together, and — more usefully — *why* each piece is the
shape it is. Almost every decision below was made or corrected by a
measurement; those are linked to `DECISIONS.md`, which is dated.

## Processes

Four, one machine, no network required on the local tier.

| | What | Why separate |
|---|---|---|
| 1 | **Model server** — `llama-server` (target) or `ollama` (today) | A model server is a different lifecycle from a daemon: it starts slowly, holds GPU memory, and should survive a `neiro` restart. |
| 2 | **`neiro`** — one asyncio daemon: capture, endpointing, STT, affect, LLM client, TTS, tools, WebSocket | Everything with a latency budget lives in one event loop, so there is one place to look when a turn is slow. |
| 3 | **The browser tab** — renders the face **and owns audio playback** | Whoever owns the audio clock owns the animation clock. Splitting them makes every viseme drift within a sentence. |
| 4 | **`neiroctl`** — a one-byte writer to a socket, bound to a key | The compositor owns the keyboard. This must not be a Python entrypoint: interpreter start-up would be most of the press-to-record latency. |

## The Turn is the spine

```python
@dataclass
class Turn:
    id: int; t0_speech_start: float; t_endpoint: float | None
    audio: np.ndarray | None; transcript: str | None; partial: str | None
    user_affect: UserAffect        # what she HEARD
    neiro_state: NeiroState        # what she FEELS
    tool_calls: list[ToolCall]
    tier: Tier                     # snapshotted at turn start, never mid-turn
    cancel: asyncio.Event          # checked at every await, from day one
    timeline: dict[str, float]     # every stage stamps this
```

Every field exists from Stage 0, inert where unused. Two of them carry
most of the design:

**`user_affect` and `neiro_state` are never assigned to each other.** One
is what we heard in his voice; the other is what she feels. Merging them
is how an assistant ends up reading its own synthesised voice back as the
user's mood. `labels.py` keeps them structurally apart: a Lane B model
predicts a point on the arousal/valence plane, never one of her six VRM
expressions.

**`cancel` is checked at every await from the first line**, not added when
barge-in lands in Stage 2. Retrofitting cancellation into a running
pipeline produces half-cancelled turns: audio stopped, LLM still
streaming, history containing words she never said.

## The flow

```
mic ─► ring(30s) ─► VAD ─► smart-turn ─► Turn ─► STT ─┐
                     │                                ├─► LLM
                     └─► prosody, on a rolling window ─┘    │
                         WHILE he is still speaking          ▼
                                                       <e:LABEL:D>
                                                             │
                       ┌─────────────────────────────────────┼──────────────┐
                       ▼                                     ▼              ▼
                  face weights                          voice instruct   next prompt
                  (blend.py)                            (voice.py)
                       │                                     │
                       └──────► browser ◄────── chunker ─► TTS
```

Affect runs *during* speech, which is why it costs the turn budget
nothing. By the time the endpoint fires, the reading already exists.

## What is measured

One number: **the endpoint event → the browser's `played` callback for
sequence 0.** p50 and p95, never a mean, per tier and per device profile.
Everything before that callback is "we sent it".

Measured stage costs on the laptop (see `DECISIONS.md` for method):

| Stage | Cost | Note |
|---|---|---|
| VAD, per 32 ms frame | 0.08 ms | CPU ONNX, 0 VRAM |
| Prosody, per 3 s window | 10 ms | during speech — free |
| STT | 141 ms | GPU, distil-large-v3.5 int8 |
| LLM | ~15 s **today** | CPU-only ollama build; the one bottleneck |
| TTS first audio | 281–1088 ms | scales with first-clause length |

## Safety

Three walls, because the threat is specific: `hyprctl dispatch` on this
machine is a Lua 5.5 eval with `os.execute` in scope, proven by
experiment.

1. **Shape, at registration time.** A bare `str` argument raises on
   import. Arguments are an `int` index into a list Neiro produced *this
   turn*, or a `Literal` enum. `extra="forbid"` is mandatory.
2. **Tier.** RED tools are not registered at all — the model cannot
   express what it cannot see. YELLOW needs a confirmation whose nonce is
   computed in the registry and never sent to the model, so the LLM does
   not participate in its own authorisation.
3. **Budget.** Per-tool sliding windows plus a global side-effect
   ceiling, so many individually-allowed actions cannot add up to a
   runaway.

The confirmation itself is two channels at once and **the click always
wins** — speech recognition hallucinates, a mouse does not.

## Tiers

`Tier` is `local` / `lan` / `tunnel`; `Locality` is `LOCAL_PINNED` /
`LAN_TIERABLE` / `TIERABLE`, and `Locality.allows(tier)` has a full
truth-table test. Capture, VAD, affect, the sink and **every tool** are
`LOCAL_PINNED`: the mic and the desktop are wherever Yash is. STT is
`LAN_TIERABLE` — shipping 16 kHz audio is fine over ethernet and breaks
the design over a tunnel. Only the LLM and TTS go anywhere.

## Where things live

```
src/neiro/
  state.py protocols.py          the contracts, written before any implementation
  config.py                      every tunable, with the measurement that set it
  affect/    features baseline prosody fusion labels null
  emotion/   blend voice         her state reaching the face and the voice
  llm/       openai_compat ollama_native emotion_tag chunker prompt
  stt/ tts/ audio/               providers behind protocols.py
  tools/     registry tiers egress confirm audit builtin hyprland system media
  training/  manifest corpora    pure-python, shared with training/ via sys.path
  orchestrator.py server.py metrics.py
web/         audio.js face.js    the audio clock and the face
training/    its own uv project, its own CUDA torch
```

## Two environments, deliberately

`training/` is a separate uv project with CUDA torch. The runtime venv
must stay CUDA-free: ctranslate2 loads cuBLAS 12 through
`LD_LIBRARY_PATH`, and a second CUDA runtime in the same process is a
failure mode already paid for once. Kokoro therefore uses CPU-index
torch.

They are not linked by a path dependency — that propagated the runtime's
CPU torch source into the training resolution and silently broke it. The
shared pure-python modules are imported via `sys.path` instead.
