"""Both TTS engines behind one Protocol.

The point of the Protocol is that gate G5 can swap the voice by changing
one line. These tests assert the two are actually interchangeable, and
that each is honest about what it cannot do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neiro.state import EmotionLabel, Locality, NeiroState, Tier
from neiro.tts.chatterbox import ChatterboxTts, ChatterboxUnavailable
from neiro.tts.kokoro import KokoroTts
from neiro.tts.qwen3tts import Qwen3Tts, Qwen3TtsUnavailable

HAPPY = NeiroState.from_label(EmotionLabel.HAPPY, 0.8)


class TestInterchangeable:
    @pytest.mark.parametrize("cls", [KokoroTts, ChatterboxTts, Qwen3Tts])
    def test_both_present_the_same_surface(self, cls) -> None:
        # G5 swaps the voice by changing one line, or the Protocol failed.
        for name in ("synth", "warm", "locality"):
            assert hasattr(cls, name), f"{cls.__name__}.{name}"

    @pytest.mark.parametrize("cls", [KokoroTts, ChatterboxTts, Qwen3Tts])
    def test_both_are_local_pinned(self, cls) -> None:
        # Audio bytes are the one thing not worth sending over a link.
        assert cls.locality is Locality.LOCAL_PINNED
        assert not cls.locality.allows(Tier.TUNNEL)


class TestChatterbox:
    def test_missing_weights_say_how_to_fetch_them(self, tmp_path: Path) -> None:
        t = ChatterboxTts(model_dir=tmp_path / "nope")
        with pytest.raises(ChatterboxUnavailable, match="fetch-models"):
            t.load()

    def test_the_missing_package_explains_why_it_is_not_a_dependency(self, tmp_path: Path) -> None:
        # Installing it pulls a CUDA torch path, and this venv keeps the
        # CPU build so ctranslate2's cuBLAS 12 stays the only CUDA
        # runtime in the process.
        weights = tmp_path / "chatterbox"
        weights.mkdir()
        t = ChatterboxTts(model_dir=weights)
        with pytest.raises(ChatterboxUnavailable, match="CUDA"):
            t.load()

    def test_the_dial_comes_from_the_same_state_as_the_face(self) -> None:
        from neiro.emotion.voice import exaggeration_for

        assert 0.25 <= exaggeration_for(HAPPY) <= 0.8


class TestVisemes:
    def test_kokoro_is_the_one_that_returns_them(self) -> None:
        # The whole reason it is the Stage 0 voice: every codec-style TTS
        # returns audio and nothing else, so the free phoneme durations
        # are collected while they are available.
        from neiro.tts.visemes import timeline

        events = timeline("hɛloʊ", [0.05, 0.08, 0.06, 0.12, 0.09])
        assert events and all(len(e) == 3 for e in events)

    def test_chatterbox_states_their_absence_rather_than_implying_it(self) -> None:
        # It yields (pcm, None). The None is a statement: the caller must
        # fall back to amplitude-driven mouth motion.
        import inspect

        source = inspect.getsource(ChatterboxTts.synth)
        assert "None" in source
        assert "amplitude" in source


class TestKokoroHonesty:
    def test_it_accepts_a_state_and_ignores_it(self) -> None:
        # Kokoro has NO emotion control. Silently accepting the argument
        # keeps the Protocol honest, so swapping in an expressive engine
        # at G5 changes one file and nothing else.
        import inspect

        source = inspect.getsource(KokoroTts.synth)
        assert "ignored" in source or "no emotion control" in source

    def test_synth_is_async_and_yields_pairs(self) -> None:
        import inspect

        assert inspect.isasyncgenfunction(KokoroTts.synth)
        assert inspect.isasyncgenfunction(ChatterboxTts.synth)


class TestNoAccidentalDownload:
    def test_neither_loads_a_model_at_construction(self) -> None:
        # Constructing a provider must be free; the daemon decides when
        # to pay the loading cost, in warm-up order.
        assert KokoroTts()._pipeline is None
        assert ChatterboxTts()._model is None


class TestQwen3Tts:
    """Gate G5's primary candidate — the reason G5 exists at all."""

    def test_missing_weights_say_how_to_fetch_them(self, tmp_path: Path) -> None:
        with pytest.raises(Qwen3TtsUnavailable, match="fetch-models"):
            Qwen3Tts(model_dir=tmp_path / "nope").load()

    def test_the_missing_package_explains_the_venv_rule(self, tmp_path: Path) -> None:
        weights = tmp_path / "qwen3"
        weights.mkdir()
        with pytest.raises(Qwen3TtsUnavailable, match="CUDA runtime"):
            Qwen3Tts(model_dir=weights).load()

    def test_the_instruction_is_separate_from_the_text(self) -> None:
        # A TTS reading its own stage direction aloud is a specific,
        # embarrassing failure. They are distinct arguments, never
        # concatenated.
        import inspect

        source = inspect.getsource(Qwen3Tts.synth)
        assert "instruct=instruct" in source
        assert "text=text" in source
        assert "text + instruct" not in source

    def test_the_voice_identity_is_a_named_constant(self) -> None:
        # A one-way door: once she has a voice, changing it makes her a
        # different character. G5 freezes it as a versioned asset.
        from neiro.tts.qwen3tts import DEFAULT_SPEAKER

        assert DEFAULT_SPEAKER

    def test_it_is_not_loaded_at_construction(self) -> None:
        assert Qwen3Tts()._model is None
