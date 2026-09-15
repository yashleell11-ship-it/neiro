"""Every provider satisfies its Protocol — by signature, not by hasattr.

The per-provider "interchangeable" tests check that a name exists. That
cannot catch the drift that actually breaks a swap: a renamed parameter
(the orchestrator calls by keyword), a method that stopped being async,
or a provider that forgot `locality`. Protocols are not runtime-checkable
for signatures, so this compares them by hand.
"""

from __future__ import annotations

import inspect

import pytest

from elizabeth import protocols
from elizabeth.affect.null import NullAffectProvider
from elizabeth.affect.prosody import ProsodyAffectProvider
from elizabeth.affect.ser import SerAffectProvider
from elizabeth.audio.endpoint import SmartTurnEndpointer
from elizabeth.audio.sink_local import LocalWavSink
from elizabeth.llm.ollama_native import OllamaNativeLlm
from elizabeth.llm.openai_compat import OpenAiCompatLlm
from elizabeth.state import Locality
from elizabeth.stt.faster_whisper import FasterWhisperStt
from elizabeth.stt.moonshine import MoonshineStt
from elizabeth.tts.chatterbox import ChatterboxTts
from elizabeth.tts.kokoro import KokoroTts
from elizabeth.tts.qwen3tts import Qwen3Tts

CASES = [
    (protocols.STT, FasterWhisperStt),
    (protocols.STT, MoonshineStt),
    (protocols.AffectProvider, NullAffectProvider),
    (protocols.AffectProvider, ProsodyAffectProvider),
    (protocols.AffectProvider, SerAffectProvider),
    (protocols.LLM, OpenAiCompatLlm),
    (protocols.LLM, OllamaNativeLlm),
    (protocols.TTS, KokoroTts),
    (protocols.TTS, ChatterboxTts),
    (protocols.TTS, Qwen3Tts),
    (protocols.Sink, LocalWavSink),
    (protocols.Endpointer, SmartTurnEndpointer),
]
IDS = [f"{p.__name__}:{i.__name__}" for p, i in CASES]


def _methods(proto: type) -> dict[str, object]:
    return {
        name: obj
        for name, obj in vars(proto).items()
        if inspect.isfunction(obj) and not name.startswith("_")
    }


def _is_async(fn: object) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def _params(fn: object) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


@pytest.mark.parametrize(("proto", "impl"), CASES, ids=IDS)
def test_every_protocol_method_exists_with_the_same_parameters(proto, impl) -> None:
    for name, ref in _methods(proto).items():
        got = getattr(impl, name, None)
        assert got is not None, f"{impl.__name__} lacks {proto.__name__}.{name}"
        assert _params(got) == _params(ref), (
            f"{impl.__name__}.{name}{inspect.signature(got)} != "
            f"{proto.__name__}.{name}{inspect.signature(ref)}"
        )


@pytest.mark.parametrize(("proto", "impl"), CASES, ids=IDS)
def test_async_methods_stay_async(proto, impl) -> None:
    # The orchestrator awaits (or `async for`s) these; a provider that
    # quietly went synchronous would raise at the call site, on a real
    # turn, never in a hasattr test.
    for name, ref in _methods(proto).items():
        if _is_async(ref) or "AsyncIterator" in str(inspect.signature(ref).return_annotation):
            assert _is_async(getattr(impl, name)), f"{impl.__name__}.{name} is not async"


@pytest.mark.parametrize(("proto", "impl"), CASES, ids=IDS)
def test_locality_is_declared_where_the_protocol_demands_it(proto, impl) -> None:
    if "locality" in getattr(proto, "__annotations__", {}):
        assert isinstance(getattr(impl, "locality", None), Locality), impl.__name__
