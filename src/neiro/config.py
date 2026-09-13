"""Every tunable in the project lives here, with a default, units, and a
comment on what it's for. Rule (CLAUDE.md): no magic number in a module
— it comes from config or it isn't tunable.

``schema_version`` exists so a later rename/restructure of this schema
can migrate an existing ``~/.config/neiro/config.toml`` instead of
silently ignoring stale keys.
"""

from __future__ import annotations

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
    dead_band_z: float = 0.5
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
