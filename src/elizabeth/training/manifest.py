"""The dataset manifest: what training data Elizabeth pulls, from where, and
under what licence.

**Why a manifest, not a shell script of wgets.** Every dataset carries a
licence that decides whether a model trained on it can ever be released
from an Apache-2.0 repo. That decision has to be recorded next to the
download, in a form a test can check, or it is lost the first time
someone adds "just one more" corpus. So `weights_publishable` is
validated against the licence flags — an NC dataset marked publishable
is a load-time error, not a surprise at release time.

Raw datasets live in `data/datasets/` (gitignored — they are large and
they are other people's voices). The manifest is committed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SCHEMA_VERSION = 1

Target = Literal[
    # --- emotion: the project's stated first priority ---
    "ser_lane_b",  # speech emotion model (Lane B)
    "emotion_deep",  # emotional speech beyond the standard four corpora
    # --- Hindi: she must understand and speak it, not only English ---
    "hindi_stt",  # Hindi + Indic speech recognition
    "hindi_tts",  # Hindi text-to-speech corpora
    "hindi_emotion_text",  # Hindi/Hinglish emotional + conversational text
    "hinglish_llm",  # code-switching + Indian context
    # --- the rest of the pipeline ---
    "stt_indian_english",  # whisper fine-tune on his accent
    "tts_voice",  # her voice, with emotion
    "persona_lora",  # her character, the <e:> tag, tool-calling
    "wakeword_speaker_noise",  # wake word, speaker verification, augmentation
    "endpointing",  # smart-turn fine-tune
]
TARGETS: tuple[str, ...] = Target.__args__  # type: ignore[attr-defined]

Flag = Literal[
    "permissive", "NC", "ND", "SA", "research-only", "request-required", "gated", "unclear"
]
Access = Literal["direct", "hf-login", "hf-gated-approval", "request-form", "paid", "unavailable"]
Publishable = Literal["yes", "no", "unclear"]

# A model trained on data carrying any of these cannot be released under
# Apache-2.0. Private use is fine; publishing the weights is not.
NON_PUBLISHABLE_FLAGS: frozenset[str] = frozenset({"NC", "ND", "research-only"})


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # directory name under data/datasets/, [a-z0-9_-]
    target: Target
    priority: int  # 1 essential, 2 good, 3 optional
    hf_id: str = ""  # exactly one of hf_id / url
    url: str = ""
    license: str
    flags: list[Flag]
    access: Access
    size_gb: float
    hours: float = 0.0  # audio hours; 0 for text-only
    weights_publishable: Publishable
    note: str = ""
    sha256: str = ""  # optional, for url downloads only
    # Glob patterns limiting WHAT is fetched from an HF repo. Without
    # this, `snapshot_download` takes the whole repository — and several
    # of these repos are multilingual mirrors where the manifest's
    # `size_gb` describes only the one language wanted. Measured the
    # expensive way on 2026-09-13: `fsicoli/common_voice_17_0` is
    # recorded at 0.5 GB for its Hindi split and reached **278 GB** of
    # every other language before it was stopped.
    allow: list[str] = []

    @field_validator("name")
    @classmethod
    def _name_is_a_safe_dirname(cls, v: str) -> str:
        # Spelled out rather than `c.isalnum() and c.islower()`: that
        # reads correctly and rejects every digit, because "5".islower()
        # is False. It is exactly the kind of quiet wrongness that only
        # shows up when a real name like "vctk-corpus-0-92" arrives.
        def ok(c: str) -> bool:
            return c.isdigit() or (c.isalpha() and c.islower() and c.isascii()) or c in "_-"

        if not v or not all(ok(c) for c in v):
            raise ValueError(f"dataset name must be [a-z0-9_-], got {v!r}")
        return v

    @field_validator("priority")
    @classmethod
    def _priority_in_range(cls, v: int) -> int:
        if not 1 <= v <= 3:
            raise ValueError(f"priority must be 1-3, got {v}")
        return v

    @field_validator("size_gb", "hours")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @model_validator(mode="after")
    def _allow_patterns_are_for_hf_only(self) -> Dataset:
        if self.allow and not self.hf_id:
            raise ValueError(f"{self.name}: `allow` patterns only apply to HF repos")
        return self

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Dataset:
        if bool(self.hf_id) == bool(self.url):
            raise ValueError(f"{self.name}: set exactly one of hf_id / url")
        if self.sha256 and self.hf_id:
            raise ValueError(f"{self.name}: sha256 is for url downloads; HF verifies its own files")
        return self

    @model_validator(mode="after")
    def _publishable_matches_flags(self) -> Dataset:
        flags = set(self.flags)
        if self.weights_publishable == "yes":
            if flags & NON_PUBLISHABLE_FLAGS:
                raise ValueError(
                    f"{self.name}: flags {sorted(flags & NON_PUBLISHABLE_FLAGS)} mean weights "
                    "trained on this cannot be released — weights_publishable must be 'no'"
                )
            if "unclear" in flags:
                raise ValueError(
                    f"{self.name}: licence is unclear, so weights_publishable cannot be 'yes'"
                )
        return self

    @property
    def is_hf(self) -> bool:
        return bool(self.hf_id)

    @property
    def source(self) -> str:
        return f"https://huggingface.co/datasets/{self.hf_id}" if self.is_hf else self.url


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    dataset: list[Dataset] = []

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"manifest schema_version {v}, this code reads {SCHEMA_VERSION}")
        return v

    @model_validator(mode="after")
    def _names_unique(self) -> Manifest:
        seen: set[str] = set()
        for d in self.dataset:
            if d.name in seen:
                raise ValueError(f"duplicate dataset name {d.name!r}")
            seen.add(d.name)
        return self

    @model_validator(mode="after")
    def _sources_unique(self) -> Manifest:
        """No two entries may point at the same download.

        Independent surveys name the same corpus differently — "svarah",
        "svarah-indian-accented-english", "svarah-indic-accented-english"
        — and a name-only check lets all three through, so the same
        bytes get fetched three times into three directories. Observed
        live: seven duplicated sources, 127.5 GB of pointless download.
        """
        seen: dict[str, str] = {}
        for d in self.dataset:
            key = (d.hf_id or d.url).lower().rstrip("/")
            if key in seen:
                raise ValueError(
                    f"{d.name!r} and {seen[key]!r} both point at {key} — "
                    "the same bytes would be downloaded twice. Keep one."
                )
            seen[key] = d.name
        return self

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(tomllib.loads(path.read_text()))

    @classmethod
    def loads(cls, text: str) -> Manifest:
        return cls.model_validate(tomllib.loads(text))

    def select(
        self,
        *,
        max_priority: int = 1,
        target: str | None = None,
        only: list[str] | None = None,
    ) -> list[Dataset]:
        """The download set. `only` wins over everything; otherwise filter
        by priority ceiling and optional target. Order: priority, then
        manifest order — so a partial run pulls the essentials first.
        """
        if only:
            unknown = set(only) - {d.name for d in self.dataset}
            if unknown:
                raise KeyError(f"not in manifest: {sorted(unknown)}")
            return [d for d in self.dataset if d.name in only]
        chosen = [d for d in self.dataset if d.priority <= max_priority]
        if target is not None:
            chosen = [d for d in chosen if d.target == target]
        return sorted(chosen, key=lambda d: d.priority)

    @staticmethod
    def total_gb(items: list[Dataset]) -> float:
        return round(sum(d.size_gb for d in items), 1)

    @staticmethod
    def total_hours(items: list[Dataset]) -> float:
        return round(sum(d.hours for d in items), 1)

    def by_target(self) -> dict[str, list[Dataset]]:
        out: dict[str, list[Dataset]] = {t: [] for t in TARGETS}
        for d in self.dataset:
            out[d.target].append(d)
        return out
