"""Both recognisers behind one Protocol.

Stage 2 A/Bs them on Yash's own voice. These assert they are actually
swappable and that each rejects silence the same way — an assistant that
answers an empty room is worse than one that says nothing.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from neiro.state import Locality, Tier
from neiro.stt.faster_whisper import FasterWhisperStt
from neiro.stt.moonshine import MoonshineStt, MoonshineUnavailable


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
