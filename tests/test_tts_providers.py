"""The TTS engines behind one Protocol.

The point of the Protocol is that gate G5 can swap the voice by changing
one line. These tests assert the three are actually interchangeable, and
that each is honest about what it cannot do.

Every engine is stood in for by a stub that records what reached it and
hands back a known buffer. Each provider's `load()` is a no-op once its
model slot is filled, so nothing here needs weights, torch or a
download. The assertions are on what a provider *does* with its engine —
the PCM it yields, the visemes it does or does not claim, the arguments
that cross the boundary — never on how its source reads. An assertion on
source text survives the regression it was written to catch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from elizabeth.emotion.voice import exaggeration_for, instruct_for
from elizabeth.state import NEUTRAL_STATE, ElizabethState, EmotionLabel, Locality, Tier
from elizabeth.tts.chatterbox import ChatterboxTts, ChatterboxUnavailable
from elizabeth.tts.kokoro import FRAMES_PER_SECOND, SAMPLERATE, KokoroTts
from elizabeth.tts.qwen3tts import DEFAULT_SPEAKER, Qwen3Tts, Qwen3TtsUnavailable
from elizabeth.tts.visemes import VISEMES

HAPPY = ElizabethState.from_label(EmotionLabel.HAPPY, 0.8)
SAD = ElizabethState.from_label(EmotionLabel.SAD, 0.2)
TEXT = "Hello there."
VOICE = "af_test"

# 10 ms at the rate every engine yields: enough to tell "audio came
# back" from "nothing did" without a real synthesis.
PCM = np.full(SAMPLERATE // 100, 0.1, dtype=np.float32)
# What kokoro's pipeline emits for "hello": IPA, plus a duration per
# phoneme in the duration head's 80-per-second frames — not seconds.
PHONEMES = "hɛloʊ"
FRAMES = [4, 6, 5, 10, 7]


class _KokoroEngine:
    """Stands in for `KPipeline`: called with `(text, voice=)`, yields
    results shaped like kokoro's — `.audio`, `.phonemes`,
    `.output.pred_dur`. `aligned=False` yields audio with no alignment,
    the shape the provider must report as `None`.
    """

    def __init__(self, aligned: bool = True) -> None:
        self.aligned = aligned
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, voice: str):
        self.calls.append((text, voice))
        output = SimpleNamespace(pred_dur=FRAMES) if self.aligned else None
        yield SimpleNamespace(audio=PCM, phonemes=PHONEMES if self.aligned else None, output=output)


class _Tensor:
    """The one thing the provider asks of a torch tensor: `.cpu()`."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> np.ndarray:
        return self._array


class _ChatterboxEngine:
    """Stands in for `ChatterboxTTS`: `generate(text, **kwargs)` returns
    a `(1, n)` tensor, the shape the real one returns."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, text: str, **kwargs) -> _Tensor:
        self.calls.append((text, kwargs))
        return _Tensor(PCM[None, :])


class _Qwen3Engine:
    """Stands in for `Qwen3TTS`: `generate_stream(**kwargs)` yields chunks."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_stream(self, **kwargs) -> list[np.ndarray]:
        self.calls.append(kwargs)
        return [PCM, PCM]


def _faked(cls):
    """A provider with its engine slot already filled, so `load()` is a
    no-op and nothing is imported or downloaded."""
    tts = cls()
    if cls is KokoroTts:
        tts._pipeline = _KokoroEngine()
    elif cls is ChatterboxTts:
        tts._model = _ChatterboxEngine()
    else:
        tts._model = _Qwen3Engine()
    return tts


async def _drain(stream):
    return [pair async for pair in stream]


def _speak(tts, text: str = TEXT, state: ElizabethState | None = HAPPY):
    return asyncio.run(_drain(tts.synth(text, state)))


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

    @pytest.mark.parametrize("cls", [KokoroTts, ChatterboxTts, Qwen3Tts])
    def test_synth_streams_float32_pcm_paired_with_visemes_or_none(self, cls) -> None:
        # The orchestrator does `async for pcm, visemes in tts.synth(...)`
        # and hands both straight to the sink; this is the shape it
        # relies on, whichever engine is behind it.
        pairs = _speak(_faked(cls))
        assert pairs
        for pcm, visemes in pairs:
            assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) == len(PCM)
            assert visemes is None or all(len(event) == 3 for event in visemes)


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

    def test_the_dial_reaches_the_engine_from_the_same_state_as_the_face(self) -> None:
        # One dial, derived from the ElizabethState that drives the face. A
        # bright expression over a flat reading is exactly the mismatch
        # people notice, so the value the engine receives must move with
        # the state, not merely exist.
        t = _faked(ChatterboxTts)
        _speak(t, state=HAPPY)
        _speak(t, state=SAD)
        _speak(t, state=None)
        (_, happy), (_, sad), (_, none) = t._model.calls
        assert happy["exaggeration"] == exaggeration_for(HAPPY)
        assert sad["exaggeration"] == exaggeration_for(SAD)
        assert happy["exaggeration"] != sad["exaggeration"]
        # No state is neutral, not a third value.
        assert none["exaggeration"] == exaggeration_for(NEUTRAL_STATE)

    def test_the_frozen_voice_reaches_the_engine_only_once_g5_picks_one(
        self, tmp_path: Path
    ) -> None:
        unpinned = _faked(ChatterboxTts)
        _speak(unpinned)
        assert "audio_prompt_path" not in unpinned._model.calls[0][1]

        pinned = ChatterboxTts(reference_wav=tmp_path / "her.wav")
        pinned._model = _ChatterboxEngine()
        _speak(pinned)
        assert pinned._model.calls[0][1]["audio_prompt_path"] == str(tmp_path / "her.wav")


class TestVisemes:
    def test_kokoro_is_the_one_that_returns_them(self) -> None:
        # The whole reason it is the Stage 0 voice: every codec-style TTS
        # returns audio and nothing else, so the free phoneme durations
        # are collected while they are available.
        ((_, visemes),) = _speak(_faked(KokoroTts))
        assert visemes and all(shape in VISEMES for _, shape, _ in visemes)
        assert visemes[0][0] == 0.0
        # pred_dur is in frames. Treating them as seconds would make the
        # mouth run eighty times longer than the audio.
        held = sum(duration for _, _, duration in visemes)
        assert held == pytest.approx(sum(FRAMES) / FRAMES_PER_SECOND)

    def test_kokoro_says_none_when_the_engine_gave_no_alignment(self) -> None:
        # The Protocol promises None for "no alignment", and it is what
        # the caller keys the amplitude fallback on.
        t = KokoroTts()
        t._pipeline = _KokoroEngine(aligned=False)
        ((pcm, visemes),) = _speak(t)
        assert len(pcm) == len(PCM)
        assert visemes is None

    @pytest.mark.parametrize("cls", [ChatterboxTts, Qwen3Tts])
    def test_the_codec_engines_state_their_absence_rather_than_implying_it(self, cls) -> None:
        # They yield (pcm, None). The None is a statement: the caller
        # must fall back to amplitude-driven mouth motion.
        pairs = _speak(_faked(cls))
        assert pairs and all(visemes is None for _, visemes in pairs)


class TestKokoroHonesty:
    def test_it_accepts_a_state_and_ignores_it(self) -> None:
        # Kokoro has NO emotion control. Silently accepting the argument
        # keeps the Protocol honest, so swapping in an expressive engine
        # at G5 changes one file and nothing else — and nothing derived
        # from the state may leak into what the engine is asked for.
        engine = _KokoroEngine()
        t = KokoroTts(voice=VOICE)
        t._pipeline = engine
        outputs = [_speak(t, state=state) for state in (None, HAPPY, SAD)]
        assert engine.calls == [(TEXT, VOICE)] * 3
        pcms = [pcm for ((pcm, _),) in outputs]
        assert all(np.array_equal(pcm, pcms[0]) for pcm in pcms)


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
        # embarrassing failure. The engine must receive the words he will
        # hear in one field and the delivery prose in another — checked
        # at the boundary, so every spelling of concatenation fails, not
        # the one a grep happened to look for.
        q = _faked(Qwen3Tts)
        _speak(q, state=HAPPY)
        (call,) = q._model.calls
        assert call["text"] == TEXT
        assert call["instruct"] == instruct_for(HAPPY)

    def test_no_state_means_no_stage_direction_at_all(self) -> None:
        # With no state there is no delivery to describe; the provider
        # sends nothing rather than inventing a neutral one.
        q = _faked(Qwen3Tts)
        _speak(q, state=None)
        assert q._model.calls[0]["instruct"] is None

    def test_the_voice_identity_is_a_named_constant(self) -> None:
        # A one-way door: once she has a voice, changing it makes her a
        # different character. G5 freezes it as a versioned asset, and it
        # is what the engine is asked for unless someone chose otherwise.
        assert DEFAULT_SPEAKER
        q = _faked(Qwen3Tts)
        _speak(q)
        assert q._model.calls[0]["speaker"] == DEFAULT_SPEAKER

    def test_it_is_not_loaded_at_construction(self) -> None:
        assert Qwen3Tts()._model is None
