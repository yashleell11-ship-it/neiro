"""Both recognisers behind one Protocol.

Stage 2 A/Bs them on Yash's own voice. These assert they are actually
swappable and that each rejects silence the same way — an assistant that
answers an empty room is worse than one that says nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from neiro.config import Neiro
from neiro.state import Locality, Tier
from neiro.stt.faster_whisper import FasterWhisperStt
from neiro.stt.moonshine import MoonshineEnglishOnly, MoonshineStt, MoonshineUnavailable

_SILENCE = np.zeros(16000, dtype=np.float32)


class TestLanguage:
    """Hindi and English are equal priorities. The recogniser learns
    which it is hearing from config, never from a literal in the module
    — the "en" that used to live there made Hindi impossible to even
    ask for.
    """

    @pytest.mark.parametrize("configured, passed", [("auto", None), ("en", "en"), ("hi", "hi")])
    def test_config_language_reaches_faster_whisper(self, configured, passed) -> None:
        seen: dict = {}

        class FakeWhisperModel:
            def transcribe(self, pcm, **kw):
                seen.update(kw)
                return iter([]), SimpleNamespace(language=passed or "hi", language_probability=0.9)

        stt = FasterWhisperStt(Neiro(stt={"language": configured}))
        stt._model = FakeWhisperModel()
        asyncio.run(stt.transcribe(_SILENCE))
        assert "language" in seen
        assert seen["language"] == passed  # None is faster-whisper's "detect it"

    def test_the_detected_language_is_kept_for_the_benchmark(self) -> None:
        class FakeWhisperModel:
            def transcribe(self, pcm, **kw):
                return iter([]), SimpleNamespace(language="hi", language_probability=0.7)

        stt = FasterWhisperStt(Neiro(stt={"language": "auto"}))
        assert stt.last_detected_language is None
        stt._model = FakeWhisperModel()
        asyncio.run(stt.transcribe(_SILENCE))
        assert stt.last_detected_language == "hi"

    def test_a_model_that_reports_no_info_does_not_crash_the_turn(self) -> None:
        class FakeWhisperModel:
            def transcribe(self, pcm, **kw):
                return iter([]), None

        stt = FasterWhisperStt(Neiro(stt={"language": "auto"}))
        stt._model = FakeWhisperModel()
        assert asyncio.run(stt.transcribe(_SILENCE)) == ""
        assert stt.last_detected_language is None

    def test_moonshine_refuses_hindi_before_the_first_utterance(self) -> None:
        # Loudly, at construction: a daemon configured for Hindi with
        # Moonshine selected fails at startup, not on his first sentence.
        with pytest.raises(MoonshineEnglishOnly, match="English-only"):
            MoonshineStt(cfg=Neiro(stt={"language": "hi"}))

    def test_the_refusal_is_one_transcribe_lets_through(self) -> None:
        # transcribe() re-raises MoonshineUnavailable and swallows every
        # other exception into "". A refusal outside that family would be
        # a silent empty turn, which is the opposite of loud.
        assert issubclass(MoonshineEnglishOnly, MoonshineUnavailable)

    @pytest.mark.parametrize("language", ["auto", "en"])
    def test_moonshine_accepts_its_one_language(self, language) -> None:
        assert MoonshineStt(cfg=Neiro(stt={"language": language}))._model is None


class TestInterchangeable:
    @pytest.mark.parametrize("cls", [FasterWhisperStt, MoonshineStt])
    def test_same_surface(self, cls) -> None:
        for name in ("transcribe", "warm", "locality"):
            assert hasattr(cls, name), f"{cls.__name__}.{name}"

    @pytest.mark.parametrize("cls", [FasterWhisperStt, MoonshineStt])
    def test_audio_may_cross_ethernet_but_never_a_tunnel(self, cls) -> None:
        # Shipping 16 kHz for every streaming partial is fine over a LAN
        # and breaks the sub-second design over a tunnel.
        assert cls.locality is Locality.LAN_TIERABLE
        assert cls.locality.allows(Tier.LAN)
        assert not cls.locality.allows(Tier.TUNNEL)

    def test_transcribe_is_async_on_both(self) -> None:
        import inspect

        assert inspect.iscoroutinefunction(FasterWhisperStt.transcribe)
        assert inspect.iscoroutinefunction(MoonshineStt.transcribe)


class TestMoonshine:
    def test_the_missing_package_explains_why_it_is_optional(self) -> None:
        # distil-whisper is the Stage 0 recogniser and already measured;
        # Moonshine only earns its place if Stage 2's A/B picks it.
        m = MoonshineStt()
        with pytest.raises(MoonshineUnavailable, match="Stage 2"):
            m.load()

    def test_it_is_not_loaded_at_construction(self) -> None:
        assert MoonshineStt()._model is None

    def test_silence_hallucinations_are_rejected(self) -> None:
        # Whisper-family models confidently transcribe an empty room as
        # "Thank you". Moonshine is a different architecture trained on
        # similar data, so the floor applies to it too rather than being
        # assumed unnecessary.
        from neiro.stt.moonshine import _HALLUCINATIONS

        for phrase in ("thank you", "you", "okay", "uh"):
            assert phrase in _HALLUCINATIONS

    def test_a_failed_utterance_returns_empty_not_a_guess(self) -> None:
        # "" is how the orchestrator recognises a rejected turn.
        class Broken(MoonshineStt):
            def load(self) -> None:
                self._model = "fake"

            def _run(self, pcm):
                raise RuntimeError("onnx exploded")

        assert asyncio.run(Broken().transcribe(np.zeros(16000, dtype=np.float32))) == ""


class TestTheTradeIsRecorded:
    def test_the_docstring_states_the_cost_rather_than_selling_it(self) -> None:
        # Moonshine is not a free win: ~269 ms after endpointing,
        # additive, traded for zero VRAM. A module that only listed the
        # upside would get chosen for the wrong reason.
        import neiro.stt.moonshine as m

        assert "VRAM" in m.__doc__
        assert "269" in m.__doc__


class TestTheEventLoopStaysFree:
    @pytest.mark.parametrize("cls", [FasterWhisperStt, MoonshineStt])
    def test_decoding_runs_off_the_event_loop(self, cls) -> None:
        # A recogniser that blocks the loop for 300 ms blocks the browser
        # socket, the affect window and the cancel check with it. Inside
        # a worker thread there is no running loop; on the loop there is.
        seen: dict[str, bool] = {}

        def record() -> None:
            try:
                asyncio.get_running_loop()
                seen["on_loop"] = True
            except RuntimeError:
                seen["on_loop"] = False

        class FakeWhisperModel:
            def transcribe(self, pcm, **_kw):
                record()
                return iter([]), None

        stt = cls()
        if cls is FasterWhisperStt:
            stt._model = FakeWhisperModel()
        else:
            stt.load = lambda: None  # type: ignore[method-assign]
            stt._model = "fake"

            def _run(pcm):
                record()
                return ""

            stt._run = _run  # type: ignore[method-assign]

        assert asyncio.run(stt.transcribe(np.zeros(16000, dtype=np.float32))) == ""
        assert seen["on_loop"] is False, f"{cls.__name__} decoded on the event loop"
