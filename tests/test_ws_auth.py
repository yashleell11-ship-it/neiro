"""The WebSocket protocol and its auth.

A localhost WebSocket is NOT private: any page the browser has open can
`new WebSocket("ws://127.0.0.1:8760/neiro")` and start listening to a
microphone-driven assistant. These tests are the wall.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neiro.server import (
    CLIENT_MESSAGES,
    PROTOCOL_VERSION,
    SERVER_MESSAGES,
    ProtocolError,
    Session,
    audio_frame,
    parse_audio_frame,
    parse_client_message,
    server_message,
)

ORIGINS = {"http://127.0.0.1:8760", "http://localhost:8760"}


class TestAuth:
    def test_the_right_token_and_origin_pass(self) -> None:
        s = Session()
        s.check(s.token, "http://127.0.0.1:8760", ORIGINS)

    def test_no_token_is_refused(self) -> None:
        s = Session()
        with pytest.raises(PermissionError):
            s.check(None, "http://127.0.0.1:8760", ORIGINS)
        with pytest.raises(PermissionError):
            s.check("", "http://127.0.0.1:8760", ORIGINS)

    def test_a_wrong_token_is_refused(self) -> None:
        s = Session()
        with pytest.raises(PermissionError):
            s.check("guessed", "http://127.0.0.1:8760", ORIGINS)

    def test_a_hostile_page_with_the_token_is_still_refused(self) -> None:
        # Defence in depth: both checks, always. A token could leak.
        s = Session()
        with pytest.raises(PermissionError, match="origin"):
            s.check(s.token, "https://evil.example", ORIGINS)

    def test_a_non_browser_client_sends_no_origin_and_is_allowed(self) -> None:
        # It cannot be a hostile web PAGE; the token protects that case.
        s = Session()
        s.check(s.token, None, ORIGINS)

    def test_tokens_differ_between_sessions(self) -> None:
        assert Session().token != Session().token

    def test_the_token_is_long_enough_not_to_be_guessed(self) -> None:
        assert len(Session().token) >= 24


class TestProtocolIsFrozen:
    def test_the_stage_0_message_set_is_complete(self) -> None:
        # Later stages fill fields; they never rename or drop a type.
        # That is what stops barge-in being rewritten in Stage 2.
        assert set(SERVER_MESSAGES) == {
            "hello",
            "utt.begin",
            "utt.chunk",
            "emotion",
            "utt.end",
            "cancel",
            "state",
        }
        assert set(CLIENT_MESSAGES) == {"ready", "played", "underrun", "error"}

    def test_an_unknown_server_message_cannot_be_sent(self) -> None:
        with pytest.raises(ProtocolError):
            server_message("utt.oops")

    def test_a_known_message_serialises_compactly(self) -> None:
        raw = server_message("utt.begin", emotion="happy", blend_ms=120)
        assert json.loads(raw) == {"t": "utt.begin", "emotion": "happy", "blend_ms": 120}
        assert " " not in raw  # no wasted bytes on the latency path

    @pytest.mark.parametrize("bad", ['{"t":"nope"}', "not json", "[]", '{"no_type":1}'])
    def test_junk_from_the_client_is_rejected(self, bad: str) -> None:
        with pytest.raises(ProtocolError):
            parse_client_message(bad)

    def test_a_valid_client_message_parses(self) -> None:
        msg = parse_client_message('{"t":"played","seq":0,"at":1.25}')
        assert msg["t"] == "played" and msg["seq"] == 0

    def test_protocol_version_is_declared(self) -> None:
        assert PROTOCOL_VERSION >= 1


class TestAudioFrames:
    def test_round_trip(self) -> None:
        pcm = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        audio_id, back = parse_audio_frame(audio_frame(7, pcm))
        assert audio_id == 7
        assert np.allclose(back, pcm)

    def test_the_id_lets_a_cancelled_reply_be_dropped(self) -> None:
        # A late chunk from a cancelled turn arriving after the next turn
        # started is otherwise a very confusing bug.
        first, _ = parse_audio_frame(audio_frame(1, np.zeros(4, dtype=np.float32)))
        second, _ = parse_audio_frame(audio_frame(2, np.zeros(4, dtype=np.float32)))
        assert first != second

    def test_an_empty_chunk_is_legal(self) -> None:
        audio_id, pcm = parse_audio_frame(audio_frame(3, np.zeros(0, dtype=np.float32)))
        assert audio_id == 3 and pcm.size == 0

    def test_a_truncated_frame_is_rejected(self) -> None:
        with pytest.raises(ProtocolError):
            parse_audio_frame(b"\x01\x02")

    def test_a_misaligned_payload_is_rejected(self) -> None:
        # Not whole float32 samples — silently reinterpreting this would
        # produce noise that sounds like a codec bug.
        with pytest.raises(ProtocolError):
            parse_audio_frame(audio_frame(1, np.zeros(4, dtype=np.float32)) + b"\x00")

    def test_it_is_binary_not_base64(self) -> None:
        # base64 on every chunk is real CPU on the latency path.
        pcm = np.zeros(240, dtype=np.float32)
        assert len(audio_frame(0, pcm)) == 4 + 240 * 4


class TestSessionState:
    def test_ready_carries_the_avatars_real_expressions(self) -> None:
        # `surprised` is often unbound in real VRM models; the blender
        # needs to know before it targets one.
        s = Session()
        assert not s.ready
        s.expressions = frozenset({"happy", "sad", "neutral"})
        s.ready = True
        assert "surprised" not in s.expressions
