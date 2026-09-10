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
    base_url: str = "http://127.0.0.1:8080"
    max_tokens: int = 160
    temperature: float = 0.7


class AffectConfig(BaseModel):
    enabled: bool = False  # flips true only after gate G3b (Task 13) passes
    dead_band_z: float = 0.5
    confidence_floor: float = 0.45
    warmup_utterances: int = 5
    baseline_window: int = 50
    ema_alpha_face: float = 0.60
    ema_alpha_words: float = 0.35


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
