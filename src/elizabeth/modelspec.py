"""Every model Elizabeth downloads: what it is, where it runs, and why.

Two things this file exists to prevent.

**Pulling 60 GB to use 2.7.** A GGUF repo holds every quantisation —
`unsloth/Qwen3.5-4B-GGUF` is 25 files, ~70 GB, and we want exactly one
of them. So a spec carries `allow` patterns and `snapshot_download`
honours them. Without this, "download the model" quietly means
"download twenty-four models you will never load".

**Confusing the inference copy with the trainable copy.** GGUF and
CTranslate2 weights cannot be fine-tuned; safetensors can. They are
different files from different repos for the same model, and the moment
you have both it is easy to point a training script at the one that
can't train. `purpose` makes that explicit and `elizabeth fetch-models
--purpose training` pulls the right half.

Sizes are recorded from the HF tree API on the date in `docs/DECISIONS.md`,
not guessed; `fetch_models.py --dry-run` re-reads the real sizes before
committing to a download.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# runtime  — loaded by the daemon to answer a turn
# training — a trainable checkpoint, only ever opened on the training box
Purpose = Literal["runtime", "training"]

# local  — the machine Yash sits at (laptop at the hostel, 8 GB VRAM)
# box    — the 3090 Ti (24 GB VRAM, 48 GB RAM, Windows), home
# both   — small enough, or needed, on either
Runs = Literal["local", "box", "both"]

Component = Literal["llm", "stt", "tts", "vad", "turn", "ser", "wake", "avatar"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # directory under models/, [a-z0-9_.-]
    component: Component
    purpose: Purpose
    runs: Runs
    hf_id: str
    license: str
    size_gb: float
    allow: list[str] = []  # glob patterns; empty = whole repo
    revision: str = "main"  # pin to a 40-char commit when a gate depends on it
    note: str = ""

    @field_validator("name")
    @classmethod
    def _safe_dirname(cls, v: str) -> str:
        if not v or not all(c.isalnum() and not c.isupper() or c in "_-." for c in v):
            raise ValueError(f"model name must be [a-z0-9_.-], got {v!r}")
        if ".." in v:
            raise ValueError(f"model name must not traverse: {v!r}")
        return v

    @field_validator("hf_id")
    @classmethod
    def _looks_like_a_repo(cls, v: str) -> str:
        if v.count("/") != 1 or not all(v.split("/")):
            raise ValueError(f"hf_id must be 'org/repo', got {v!r}")
        return v

    @model_validator(mode="after")
    def _training_weights_are_not_quantised_dead_ends(self) -> ModelSpec:
        # GGUF and CTranslate2 weights cannot be fine-tuned. Catching this
        # here beats discovering it after a 17 GB download and an hour of
        # writing a training script around the wrong file.
        if self.purpose == "training":
            frozen = [p for p in self.allow if "gguf" in p.lower()]
            if frozen or "-ct2" in self.hf_id.lower() or "gguf" in self.hf_id.lower():
                raise ValueError(
                    f"{self.name}: purpose='training' but the weights are quantised for "
                    "inference only (GGUF/CT2). Point at the safetensors repo instead."
                )
        return self

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.hf_id}"


# The GGUF quant choice, and why, for both tiers:
#
#   laptop  Qwen3.5-4B-Q4_K_M         2.74 GB — the plan's pick; leaves room
#                                      for STT + TTS + the browser in 7730 MiB
#   box     Qwen3.6-35B-A3B-UD-Q3_K_XL 16.85 GB — fits beside a 3.9 GB expressive
#                                      TTS in 24 GB, with KV to spare
#   box     Qwen3.6-35B-A3B-UD-Q4_K_M  22.13 GB — does NOT fit beside TTS in VRAM,
#                                      but the box has 48 GB of RAM and this is a
#                                      3B-active MoE, so partial CPU offload is
#                                      cheap. Downloaded to be A/B'd on the box,
#                                      not assumed better.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="qwen3.5-4b-gguf",
        component="llm",
        purpose="runtime",
        runs="local",
        hf_id="unsloth/Qwen3.5-4B-GGUF",
        license="apache-2.0",
        size_gb=2.74,
        allow=["Qwen3.5-4B-Q4_K_M.gguf"],
        note="the hostel tier's brain; also the tier that must work with no network at all",
    ),
    ModelSpec(
        name="qwen3.6-35b-a3b-q3",
        component="llm",
        purpose="runtime",
        runs="box",
        hf_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        license="apache-2.0",
        size_gb=16.85,
        allow=["Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"],
        note="the home tier; sized to leave VRAM for an expressive TTS",
    ),
    ModelSpec(
        name="qwen3.6-35b-a3b-q4",
        component="llm",
        purpose="runtime",
        runs="box",
        hf_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        license="apache-2.0",
        size_gb=22.13,
        allow=["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
        note="A/B candidate against q3 on the box's 48 GB RAM; MoE, only 3B active",
    ),
    ModelSpec(
        name="qwen3.5-4b-safetensors",
        component="llm",
        purpose="training",
        runs="box",
        hf_id="Qwen/Qwen3.5-4B",
        license="apache-2.0",
        size_gb=9.34,
        note="the LoRA base — her persona, the <e:> tag, and tool-calling are trained onto this",
    ),
    ModelSpec(
        name="distil-large-v3.5-ct2",
        component="stt",
        purpose="runtime",
        runs="both",
        hf_id="distil-whisper/distil-large-v3.5-ct2",
        license="mit",
        size_gb=1.52,
        note="already downloaded and proven by Gate G1 on sm_120",
    ),
    ModelSpec(
        name="large-v3-turbo-ct2",
        component="stt",
        purpose="runtime",
        runs="both",
        hf_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        revision="4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
        license="mit",
        size_gb=1.62,
        note=(
            "the Hindi half of the project. distil-large-v3.5 carries a multilingual "
            "vocabulary but an English-trained decoder: pinned to hi it scores 100% WER "
            "on Kathbath and the confidence floors reject 76 of 120 clips, and on auto it "
            "confidently TRANSLATES Hindi speech into English that passes the floors, "
            "which is worse. Whisper large-v3-turbo is genuinely multilingual, 809M, and "
            "4 decoder layers rather than 32, so it is the cheapest honest candidate."
        ),
    ),
    ModelSpec(
        name="large-v3-turbo",
        component="stt",
        purpose="training",
        runs="both",
        hf_id="openai/whisper-large-v3-turbo",
        revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        license="mit",
        size_gb=1.62,
        note=(
            "safetensors twin of the ct2 runtime copy, for the Hindi fine-tune on "
            "Kathbath + IndicVoices; converted back to ct2 afterwards"
        ),
    ),
    ModelSpec(
        name="distil-large-v3.5",
        component="stt",
        purpose="training",
        runs="box",
        hf_id="distil-whisper/distil-large-v3.5",
        license="mit",
        size_gb=3.04,
        note="safetensors twin of the ct2 runtime copy; fine-tuned on his accent, then converted back",
    ),
    ModelSpec(
        name="kokoro-82m",
        component="tts",
        purpose="runtime",
        runs="both",
        hf_id="hexgrad/Kokoro-82M",
        license="apache-2.0",
        size_gb=0.36,
        note="Stage 0 plumbing voice — no emotion control at all, and free viseme timings",
    ),
    ModelSpec(
        name="qwen3-tts-1.7b",
        component="tts",
        purpose="runtime",
        runs="box",
        hf_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        license="apache-2.0",
        size_gb=4.52,
        note="Gate G5 candidate: free-form instruct string is the expressive-voice bet",
    ),
    ModelSpec(
        name="chatterbox",
        component="tts",
        purpose="runtime",
        runs="box",
        hf_id="ResembleAI/chatterbox",
        license="mit",
        size_gb=13.87,
        note="G5 fallback; whole repo is large because it ships multilingual checkpoints",
    ),
    ModelSpec(
        name="openwakeword-features",
        component="wake",
        purpose="runtime",
        runs="both",
        hf_id="littlebearlabs/openwakeword-features",
        license="apache-2.0",
        size_gb=0.01,
        revision="5e032d9ecdb798f9182ca8088284cf934f10d68e",
        allow=["*.onnx", "LICENSE", "README.md"],
        note=(
            "Front end for 'Hey Elizabeth': melspectrogram.onnx then "
            "embedding_model.onnx, 80 ms of audio to one 96-d vector. Only the "
            "feature extractors -- the wake head itself is trained here, so "
            "this is the one third-party piece. NOT the openwakeword package: "
            "its Linux dependency tflite-runtime stops at cp311 and this "
            "project is 3.12, so the package cannot install without --no-deps. "
            "Plain onnxruntime runs these two files. NOT davidscripka/"
            "openwakeword either, which is cc-by-nc-sa-4.0 on the Hub and "
            "carries no weights; this mirror is apache-2.0 with a LICENSE file."
        ),
    ),
    ModelSpec(
        name="silero-vad",
        component="vad",
        purpose="runtime",
        runs="both",
        hf_id="onnx-community/silero-vad",
        license="mit",
        size_gb=0.01,
        note="Stage 2 endpointing trigger; 32 ms frames",
    ),
    ModelSpec(
        name="smart-turn-v3",
        component="turn",
        purpose="runtime",
        runs="both",
        hf_id="pipecat-ai/smart-turn-v3",
        license="bsd-2-clause",
        size_gb=0.09,
        note="decides end-of-turn so 'I want to go to... uh... the library' is not cut",
    ),
    ModelSpec(
        name="w2v-bert-2.0",
        component="ser",
        purpose="training",
        runs="box",
        hf_id="facebook/w2v-bert-2.0",
        license="mit",
        size_gb=4.65,
        note="Lane B speech-emotion encoder; fine-tuned on the emotion corpora in data/datasets.toml",
    ),
)


def by_name(name: str) -> ModelSpec:
    for m in MODELS:
        if m.name == name:
            return m
    raise KeyError(f"no model {name!r}; have {[m.name for m in MODELS]}")


def select(
    *,
    purpose: str | None = None,
    runs: str | None = None,
    component: str | None = None,
    only: list[str] | None = None,
) -> list[ModelSpec]:
    """Filter the spec list. `runs` matches 'both' as well as the exact
    tier — a model that runs on either machine is wanted by both.
    """
    if only:
        unknown = set(only) - {m.name for m in MODELS}
        if unknown:
            raise KeyError(f"not in modelspec: {sorted(unknown)}")
        return [m for m in MODELS if m.name in only]
    out = list(MODELS)
    if purpose:
        out = [m for m in out if m.purpose == purpose]
    if component:
        out = [m for m in out if m.component == component]
    if runs:
        out = [m for m in out if m.runs in (runs, "both")]
    return out


def total_gb(items: list[ModelSpec]) -> float:
    return round(sum(m.size_gb for m in items), 2)
