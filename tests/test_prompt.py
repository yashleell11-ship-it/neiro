"""Tests for the character prompt and its loader.

Two kinds of test here. The loader tests protect the KV-cache property
(stable bytes, nothing interpolated per turn). The content tests protect
the character itself — the plan's review found every candidate design
left the character unbudgeted, and a prompt that quietly loses its rules
is how she becomes a generic assistant with a face.
"""

from __future__ import annotations

import hashlib

from neiro.llm.prompt import (
    PROMPT_VERSION,
    PROMPTS_DIR,
    load_prompt,
    prompt_fingerprint,
    system_message,
    user_message,
)


class TestLoader:
    def test_prompt_loads(self) -> None:
        assert load_prompt().strip()

    def test_bytes_are_the_file_on_disk_every_turn(self) -> None:
        # The KV prefix cache depends on this being byte-identical every
        # turn — a drifting prompt is a silent latency cliff. Compare
        # against the file read independently: `load_prompt() ==
        # load_prompt()` cannot fail (it is lru_cached) and proved
        # nothing about interpolation.
        on_disk = (PROMPTS_DIR / f"{PROMPT_VERSION}.md").read_text().strip()
        assert system_message()["content"] == on_disk
        user_message("yeah, I'm fine", voice_annotation="energy +9.9σ")  # a turn happened
        assert system_message()["content"] == on_disk
        assert prompt_fingerprint() == hashlib.sha256(on_disk.encode()).hexdigest()[:12]

    def test_system_message_shape(self) -> None:
        msg = system_message()
        assert msg["role"] == "system"
        assert msg["content"] == load_prompt()

    def test_fingerprint_is_short_and_stable(self) -> None:
        fp = prompt_fingerprint()
        assert len(fp) == 12
        assert fp == prompt_fingerprint(PROMPT_VERSION)


class TestUserMessage:
    def test_plain_transcript(self) -> None:
        msg = user_message("what's my battery at")
        assert msg["role"] == "user"
        assert msg["content"] == "what's my battery at"

    def test_annotation_goes_on_the_user_turn(self) -> None:
        # This turn's actual values must live on the user message and
        # never get baked into the system prompt — mood in the system
        # prompt bleeds across every later turn and turns her into a
        # caricature of whatever she was told once.
        #
        # (The system prompt does contain a [voice: ...] EXAMPLE, which
        # is correct and necessary — she has to know what the annotation
        # means. So this uses a value that couldn't appear there.)
        this_turn = "energy +9.9σ, pitch -7.7σ"
        msg = user_message("yeah, I'm fine", voice_annotation=this_turn)
        assert f"[voice: {this_turn}]" in msg["content"]
        assert "yeah, I'm fine" in msg["content"]
        assert this_turn not in system_message()["content"]

    def test_no_annotation_means_no_marker_at_all(self) -> None:
        # Omitted, not softened. "He sounds normal" every turn is noise
        # she will eventually act on.
        assert "[voice" not in user_message("hello")["content"]


class TestCharacterRules:
    """The prompt is a real artifact with requirements, not prose."""

    def test_specifies_the_emotion_tag_format(self) -> None:
        prompt = load_prompt()
        assert "<e:" in prompt
        for label in ("happy", "angry", "sad", "relaxed", "surprised", "neutral"):
            assert label in prompt

    def test_forbids_markup_that_would_be_read_aloud(self) -> None:
        prompt = load_prompt().lower()
        for rule in ("markdown", "emoji", "bullet"):
            assert rule in prompt, f"prompt should forbid {rule}"

    def test_asks_for_a_short_opening_clause(self) -> None:
        # Gate G6: TTS time-to-first-audio scales with chunk duration, so
        # a short opener is worth 200-400ms on every single turn.
        assert "short opening clause" in load_prompt().lower()

    def test_states_words_outrank_tone(self) -> None:
        prompt = load_prompt().lower()
        assert "outrank" in prompt

    def test_caps_noticing_tone_at_once(self) -> None:
        assert "once" in load_prompt().lower()

    def test_tells_her_to_admit_not_knowing(self) -> None:
        prompt = load_prompt().lower()
        assert "don't know" in prompt or "no idea" in prompt

    def test_is_short_enough_to_stay_cheap_to_prefill(self) -> None:
        # Every token here is prefilled on a cache miss and sits in
        # context on every turn. Long character prompts are where
        # latency budgets quietly go.
        assert len(load_prompt().split()) < 700
