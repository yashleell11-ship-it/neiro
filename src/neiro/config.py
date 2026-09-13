"""Every tunable in the project lives here, with a default, units, and a
comment on what it's for. Rule (CLAUDE.md): no magic number in a module
— it comes from config or it isn't tunable.

``schema_version`` exists so a later rename/restructure of this schema
can migrate an existing ``~/.config/neiro/config.toml`` instead of
silently ignoring stale keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class AudioDeviceProfile(BaseModel):
    """One named device pair. Never use PipeWire's 'default' — see
    Stage 0 Task 0: this laptop's default sink/source silently changes
    (earbuds vs built-in), which is exactly the failure mode a voice
    assistant must not inherit.
    """

    input_device: str | None = None  # PortAudio device string; None = unset
    output_device: str | None = None
    node_name_hint: str = ""  # informational, for `neiro doctor --audio-inventory`


class AudioConfig(BaseModel):
    active_profile: str = ""  # "earbuds" | "speakers"; empty = not yet configured
    profiles: dict[str, AudioDeviceProfile] = Field(default_factory=dict)
    input_samplerate: int = 16000
    input_blocksize: int = 512  # exactly one 32 ms Silero window @16 kHz
    output_samplerate: int = 24000
    output_blocksize: int = 240  # 10 ms @24 kHz
    pre_roll_s: float = 0.3  # Task 4: audio already in the ring before you
    # press the key, prepended so the first syllable is never clipped
    ring_buffer_seconds: float = 2.0  # Task 4's terminal PTT only needs a
    # couple of seconds of pre-roll headroom; Stage 1's real endpointer
    # widens this — see docs/ARCHITECTURE.md's "30s ring, Whisper needs
    # the whole utterance" note


class SttConfig(BaseModel):
    model_id: str = "distil-whisper/distil-large-v3.5-ct2"
    compute_type: str = "int8_float16"
    beam_size: int = 1
    condition_on_previous_text: bool = False
    no_speech_prob_floor: float = 0.4
    avg_logprob_floor: float = -1.0
    # What language he is speaking. Hindi and English are equal
    # priorities (docs/DECISIONS.md, 2026-09-13) and he switches between
    # them mid-sentence, so the default lets the recogniser detect per
    # utterance instead of pinning one for the whole session. Pin it when
    # the DETECTOR is what is failing — Hinglish tagged as Urdu, a short
    # Hindi command tagged as English — and let `scripts/bench_stt.py
    # --language` show whether pinning helped, on a corpus, before
    # trusting it on his voice. faster-whisper gets `language=None` for
    # "auto"; Moonshine is English-only and refuses "hi" outright rather
    # than returning English-shaped nonsense for Hindi audio.
    language: Literal["auto", "en", "hi"] = "auto"


class LlmConfig(BaseModel):
    # The only difference between the local tier and the 3090 Ti tiers
    # (lan / tunnel): the box is one machine reached over two links.
    # ollama's OpenAI-compatible endpoint is on 11434; llama-server on 8080.
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    max_tokens: int = 160  # a voice reply, not an essay
    temperature: float = 0.7


class AffectConfig(BaseModel):
    enabled: bool = False  # flips true only after gate G3b (Task 13) passes
    # 2.0, not the plan's 0.5. Measured on 75 held-out NEUTRAL utterances
    # from 25 CREMA-D speakers, each scored against that speaker's own
    # other neutral clips — i.e. the false-positive rate on genuinely
    # ordinary speech:
    #
    #     dead-band   ordinary speech flagged   angry speech caught
    #        0.50               61%                     96%   <- the plan's value
    #        1.00               39%                     95%
    #        1.50               19%                     91%
    #        2.00                4%                     88%
    #        2.50                0%                     85%
    #
    # At 0.5σ she would remark on his tone two turns in three, while her
    # own prompt says "most of the time, say nothing about it at all".
    # Trading 8 points of recall to cut false positives from 61% to 4%
    # is the whole difference between a character who notices something
    # and one who is insufferable.
    dead_band_z: float = 2.0
    confidence_floor: float = 0.45
    warmup_utterances: int = 5
    baseline_window: int = 50
    ema_alpha_face: float = 0.60
    ema_alpha_words: float = 0.35

    # --- feature extraction (affect/features.py) ---
    # "yin" + an energy gate, chosen by measurement on 60 real clips —
    # see the table in affect/features.py::_f0. pyin marked most real
    # short-utterance speech as UNVOICED (24-28 of 60 usable) at 20x the
    # cost; yin was 60/60 at correct pitch. "pyin" stays available for
    # long clean single-speaker audio, where its voicing is better.
    f0_algorithm: str = "yin"
    f0_min_hz: float = 60.0  # below a typical adult male floor
    f0_max_hz: float = 400.0  # above a typical adult female ceiling
    frame_length: int = 1024  # 64 ms at 16 kHz — the ENERGY analysis frame
    # Pitch needs a much longer window than energy does. librosa's own
    # requirement for pyin is frame_length >= 4 * sr / fmin, which at
    # fmin=60 and 16 kHz is 1067 — so the 1024 used for energy is just
    # under it. Measured consequence on real speech (CREMA-D,
    # 2026-09-13): at 1024 pyin returned ZERO voiced frames for a normal
    # male speaking voice; at 2048 the same clip gave 36 voiced frames at
    # 122.8 Hz. 2048 = 128 ms, about 15 periods at 120 Hz.
    f0_frame_length: int = 2048
    hop_length: int = 256  # 16 ms
    # 0 = trust pyin's own voiced_flag, which is what you want. This is
    # NOT a 0-1 confidence: it is posterior mass on the winning pitch
    # candidate, ~0.15 at most on real speech. A 0.5 floor here
    # rejected 100% of real utterances while passing synthetic tones.
    voiced_prob_floor: float = 0.0  # unused by the yin path; see _f0
    # A frame is "voiced" if it is within this many dB of the clip's own
    # peak AND yin found a pitch in band. Relative, not absolute dBFS:
    # mic gain and distance move absolute level far more than the effect
    # being measured.
    voiced_db_below_peak: float = 25.0
    silence_db_below_peak: float = 35.0  # a frame this far under peak is a pause

    # --- Lane B, the trained model (affect/ser.py) ---
    # Off until gate G3b passes. Measured 2026-09-13 on this laptop:
    # Lane B costs ~590 ms per window on CPU against Lane A's ~10 ms —
    # 59x, and against a 750 ms interval that leaves almost no headroom
    # while STT and the browser are also running. So Lane B gets its own,
    # longer interval, and `lane_b_device` exists for the box tier where
    # VRAM is not scarce.
    lane_b_enabled: bool = False
    lane_b_interval_s: float = 1.5
    lane_b_device: str = "cpu"
    # Deliberately no clip-length knob. The length every training
    # example was padded or cropped to is recorded in the checkpoint
    # (`args.seconds`), and affect/ser.py fits the live window to THAT
    # with the recipe's own function. A value here could disagree with
    # the weights, and the weights would not say so.

    # --- window (affect/prosody.py) ---
    window_seconds: float = 3.0  # rolling analysis window
    min_window_seconds: float = 0.7  # shorter than this says nothing at all
    min_voiced_ratio: float = 0.15  # mostly-silence windows carry no prosody

    # --- baseline (affect/baseline.py) ---
    # Robust statistics, not mean/std: one shout must not move "his
    # normal" for the rest of the day. MAD * 1.4826 estimates sigma for
    # normally distributed data.
    mad_to_sigma: float = 1.4826
    min_sigma_fraction: float = 0.05  # floor on sigma, as a fraction of |median|
    drift_z_threshold: float = 4.0  # this far out, this consistently, = a new voice/device
    drift_consecutive: int = 8  # ...for this many utterances in a row
    max_abs_z: float = 6.0  # clamp; beyond this the feature is broken, not expressive


class VadConfig(BaseModel):
    """Silero VAD — the Stage 2 endpointing trigger and the barge-in gate.

    Silero v5 is strict about frame size: 512 samples at 16 kHz, and
    nothing else. It is a stateful RNN, so frames must be fed in order
    and the state carried between them.
    """

    model_path: str = "models/silero-vad/onnx/model.onnx"
    frame_samples: int = 512  # 32 ms at 16 kHz. Not negotiable — see vad.py.
    threshold: float = 0.5  # speech probability to call a frame speech
    # Hysteresis, so one noisy frame neither starts nor ends an utterance.
    # Leaving is deliberately slower than entering: cutting someone off
    # mid-sentence is much worse than a little trailing silence.
    min_speech_ms: float = 96.0
    min_silence_ms: float = 200.0
    # Barge-in is armed only after this much of her own speech has played,
    # so the first syllable of her reply cannot interrupt her.
    bargein_dead_zone_ms: float = 200.0
    bargein_speech_ms: float = 250.0
    bargein_probability: float = 0.85  # stricter than `threshold` on purpose


class EndpointConfig(BaseModel):
    """smart-turn v3.2 — deciding whether he actually finished a TURN.

    VAD says "he stopped making noise". That is not the same question,
    and conflating them is what cuts people off at "I want to go
    to... uh...". VAD is the cheap trigger; this is the decision.
    """

    model_path: str = "models/smart-turn-v3/smart-turn-v3.2-cpu.onnx"
    # The model takes an 80-bin log-mel over exactly 8 seconds at 16 kHz
    # (80 x 800 frames). Shorter audio is left-padded; longer is
    # truncated to the most RECENT 8 s, because the end of an utterance
    # is what decides whether it ended.
    context_seconds: float = 8.0
    n_mels: int = 80
    n_frames: int = 800
    # Calibrated on pipecat's own labelled set (smart-turn-human-5, 200
    # endpoints + 200 non-endpoints), 2026-09-13 — AUC 0.996:
    #
    #     threshold   accuracy   endpoints caught   FALSE CUTS
    #        0.60       0.993         0.995            1.0%
    #        0.70       0.990         0.985            0.5%
    #        0.72       0.968         0.940            0.5%
    #
    # 0.70, not 0.60: halving false cuts for one point of recall is the
    # right trade when cutting someone off mid-sentence costs the whole
    # turn and being a moment slow costs a moment. Same asymmetry as the
    # speech gate and the expression blender.
    complete_threshold: float = 0.70
    # Hard stop, so a model that never fires cannot hang the
    # conversation. The plan's value.
    max_wait_s: float = 2.5


class ExpressionConfig(BaseModel):
    """How her face moves. Every number here is visible to a human eye —
    these are the difference between "alive" and "a mask that snaps".
    """

    # Asymmetric on purpose: expressions arrive faster than they leave.
    # A face that fades in and out at the same rate reads as mechanical;
    # real faces light up quickly and settle slowly.
    tau_rise_s: float = 0.12
    tau_fall_s: float = 0.30

    # Surprise is physiologically brief. Held much past a second it stops
    # reading as surprise and starts reading as a stare.
    surprised_hold_s: float = 0.9
    surprised_decay_s: float = 0.6

    # VRM expression weights are additive on the same mesh; letting them
    # sum past 1 produces geometry that looks broken rather than
    # expressive. Scaled down together, so the mix is preserved.
    max_total_weight: float = 1.0

    # Below this a weight is treated as zero — stops a long exponential
    # tail leaving 0.003 of "angry" on her face all evening.
    epsilon: float = 0.01


class Neiro(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file="~/.config/neiro/config.toml",
        env_prefix="NEIRO_",
    )

    schema_version: int = 1
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    affect: AffectConfig = Field(default_factory=AffectConfig)
    expression: ExpressionConfig = Field(default_factory=ExpressionConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Without this, `toml_file` in model_config is inert — pydantic-settings
        # warns "Config key `toml_file` is set ... but will be ignored" and
        # `neiro doctor` never actually reads ~/.config/neiro/config.toml,
        # which is exactly why Task 0's "audio device profiles named" check
        # kept failing even after config.toml was written by hand.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
