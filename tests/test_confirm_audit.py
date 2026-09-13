"""The confirmation gate and the audit log.

This is the one place where being wrong changes the machine, so the
rules are strict and each has a test: the click beats the voice,
ambiguity is no, silence is no, and a sentence is not a confirmation.
"""

from __future__ import annotations

import json

import pytest

from neiro.config import SttConfig
from neiro.tools.audit import ATTEMPTED, SUCCEEDED, AuditLog
from neiro.tools.confirm import (
    HALLUCINATIONS,
    MAX_WORDS,
    NO,
    YES,
    Answer,
    NotificationConfirmer,
    decide,
    interpret,
    normalise,
)


class TestInterpretingSpeech:
    @pytest.mark.parametrize("said", ["yes", "Yeah", "go ahead", "do it", "sure", "yep"])
    def test_a_clear_yes_is_a_yes(self, said: str) -> None:
        assert interpret(said).allowed

    @pytest.mark.parametrize("said", ["okay", "Okay.", "OK", "ha"])
    def test_okay_is_deliberately_not_a_yes(self, said: str) -> None:
        """The most natural English agreement is also Whisper's commonest
        silence hallucination.

        An empty room transcribes as "Okay." often enough that this test
        caught it: "okay." normalises to "okay", which was in the YES
        lexicon, so hallucinated silence would have authorised a YELLOW
        action. The confidence floor catches most of those — but a safety
        mechanism must not rest on one check, and there are plenty of
        other natural ways to say yes that are not what silence sounds
        like. ("ha" went too; it is a laugh.)
        """
        assert not interpret(said).allowed

    @pytest.mark.parametrize("said", ["haan", "haan ji", "theek hai", "kar do"])
    def test_hindi_and_hinglish_yes(self, said: str) -> None:
        # This is how he actually answers. Requiring English for the
        # safety mechanism specifically would be the wrong place to.
        assert interpret(said).allowed

    @pytest.mark.parametrize("said", ["no", "nope", "stop", "cancel", "nahi", "rehne do", "ruko"])
    def test_a_clear_no_is_a_no(self, said: str) -> None:
        answer = interpret(said)
        assert answer.decision == "no" and not answer.allowed

    @pytest.mark.parametrize("said", ["", "   ", "uh", "hmm", "maybe", "what", "i guess"])
    def test_ambiguity_is_not_permission(self, said: str) -> None:
        # An assistant that acts on "uh" is worse than one that asks
        # twice.
        assert not interpret(said).allowed

    @pytest.mark.parametrize("said", sorted(HALLUCINATIONS))
    def test_whisper_silence_hallucinations_never_mean_yes(self, said: str) -> None:
        # These are what an empty room transcribes as.
        assert not interpret(said).allowed

    def test_a_sentence_is_not_a_confirmation(self) -> None:
        # Substring matching is how "no, don't do that" becomes a yes.
        assert not interpret("yes I think we should probably do that").allowed
        assert not interpret("no don't do that please stop").allowed

    def test_the_word_cap_is_tight(self) -> None:
        assert MAX_WORDS <= 4

    def test_low_stt_confidence_is_not_permission(self) -> None:
        # Re-checked here, not merely trusted from upstream: this is
        # where being wrong changes the machine.
        assert not interpret("yes", no_speech_prob=0.9).allowed
        assert not interpret("yes", avg_logprob=-2.0).allowed
        assert interpret("yes", no_speech_prob=0.1, avg_logprob=-0.3).allowed

    def test_tightening_the_config_tightens_the_gate(self) -> None:
        # The gate once carried its own copy of the floors, so tightening
        # them in config made transcription stricter and left the one
        # check that gates side effects exactly where it was. Now the
        # same `cfg.stt` feeds both, and a "yes" that clears the default
        # floor is refused the moment the floor moves.
        loose = SttConfig()
        assert interpret("yes", no_speech_prob=0.3, stt=loose).allowed
        assert not interpret(
            "yes", no_speech_prob=0.3, stt=SttConfig(no_speech_prob_floor=0.2)
        ).allowed
        assert interpret("yes", avg_logprob=-0.8, stt=loose).allowed
        assert not interpret("yes", avg_logprob=-0.8, stt=SttConfig(avg_logprob_floor=-0.5)).allowed

    def test_the_gate_has_no_floor_of_its_own(self) -> None:
        # With nothing passed, the boundary is config's default to the
        # third decimal, in both directions: there is one definition of
        # each floor, and it is not in confirm.py.
        floors = SttConfig()
        assert interpret("yes", no_speech_prob=floors.no_speech_prob_floor - 0.001).allowed
        assert not interpret("yes", no_speech_prob=floors.no_speech_prob_floor + 0.001).allowed
        assert interpret("yes", avg_logprob=floors.avg_logprob_floor + 0.001).allowed
        assert not interpret("yes", avg_logprob=floors.avg_logprob_floor - 0.001).allowed

    def test_punctuation_and_case_do_not_matter(self) -> None:
        assert interpret("Yes!").allowed
        assert interpret("  YES.  ").allowed

    def test_the_lexicons_do_not_overlap(self) -> None:
        assert not (YES & NO)

    def test_normalise_strips_punctuation_but_keeps_apostrophes(self) -> None:
        assert normalise("Don't!") == "don't"


class TestTwoChannels:
    def test_the_click_wins_over_the_voice(self) -> None:
        # A mouse does not hallucinate. A voice "yes" against a clicked
        # "no" must resolve to no.
        voice_yes = Answer("yes", "voice")
        click_no = Answer("no", "click")
        assert decide(voice_yes, click_no).decision == "no"

    def test_a_click_yes_beats_a_voice_no_too(self) -> None:
        # Symmetric, deliberately: the click is authoritative, not
        # merely safer.
        assert decide(Answer("no", "voice"), Answer("yes", "click")).decision == "yes"

    def test_the_voice_decides_when_nothing_was_clicked(self) -> None:
        assert decide(Answer("yes", "voice"), None).allowed
        assert decide(Answer("yes", "voice"), Answer("timeout", "timeout")).allowed

    def test_nothing_from_either_channel_is_a_no(self) -> None:
        assert not decide(None, None).allowed
        assert not decide(Answer("ambiguous", "voice"), Answer("timeout", "timeout")).allowed

    def test_timeout_is_deny_not_retry(self) -> None:
        assert decide(None, None).decision == "timeout"


class TestNotification:
    def test_a_clicked_yes_is_read_from_stdout(self) -> None:
        class FakeRun:
            def __call__(self, cmd, **kw):
                assert "--action=yes=Yes" in cmd
                assert "--wait" in cmd
                return type("R", (), {"stdout": "yes\n"})()

        assert NotificationConfirmer(runner=FakeRun()).ask("Set volume to 30?").decision == "yes"

    def test_a_dismissed_notification_is_a_timeout_not_a_yes(self) -> None:
        class FakeRun:
            def __call__(self, cmd, **kw):
                return type("R", (), {"stdout": ""})()

        assert not NotificationConfirmer(runner=FakeRun()).ask("?").allowed

    def test_a_missing_notify_send_does_not_become_permission(self) -> None:
        # No desktop, no notification — and certainly no yes.
        def boom(cmd, **kw):
            raise FileNotFoundError("notify-send")

        assert not NotificationConfirmer(runner=boom).ask("?").allowed


class TestAuditLog:
    def test_an_attempt_is_written_before_the_outcome(self, tmp_path) -> None:
        # A log written only after success cannot answer "did it run?"
        log = AuditLog(path=tmp_path / "audit.jsonl")
        first = log.attempt("set_volume", {"percent": 30}, "yellow", 1)
        log.succeeded("set_volume", 1, "volume 30 percent", call=first["call"])
        events = [e["event"] for e in log.entries]
        assert events == [ATTEMPTED, SUCCEEDED]

    def test_an_unfinished_attempt_is_findable(self, tmp_path) -> None:
        # A tool that hung or crashed the daemon leaves only this line,
        # and that gap is the evidence.
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.attempt("focus_window", {"index": 2}, "yellow", 1)
        done = log.attempt("set_volume", {"percent": 30}, "yellow", 2)
        log.succeeded("set_volume", 2, call=done["call"])
        unfinished = log.unfinished()
        assert len(unfinished) == 1
        assert unfinished[0]["tool"] == "focus_window"

    def test_the_same_tool_twice_in_a_turn_is_not_falsely_closed(self, tmp_path) -> None:
        # The model calls focus_window twice in one turn; the first
        # returns, the second hangs on the Hyprland socket and the daemon
        # is killed. Keyed on (tool, turn) alone, the one success closed
        # both attempts and the hung call left no trace at all. One
        # outcome closes exactly one attempt.
        log = AuditLog(path=tmp_path / "audit.jsonl")
        first = log.attempt("focus_window", {"index": 0}, "yellow", 7)
        log.succeeded("focus_window", 7, call=first["call"])
        hung = log.attempt("focus_window", {"index": 1}, "yellow", 7)
        unfinished = log.unfinished()
        assert len(unfinished) == 1
        assert unfinished[0]["call"] == hung["call"]
        assert unfinished[0]["args"] == {"index": 1}

    def test_an_outcome_closes_its_own_attempt_not_the_earliest(self, tmp_path) -> None:
        # Two calls in flight and the SECOND finishes first. Pairing
        # outcomes with attempts by count would report the right number
        # of hung calls with the wrong arguments, and the arguments are
        # what you read at three in the morning.
        log = AuditLog(path=tmp_path / "audit.jsonl")
        first = log.attempt("focus_window", {"index": 0}, "yellow", 7)
        second = log.attempt("focus_window", {"index": 1}, "yellow", 7)
        log.failed("focus_window", 7, "socket timeout", call=second["call"])
        unfinished = log.unfinished()
        assert [e["call"] for e in unfinished] == [first["call"]]
        assert unfinished[0]["args"] == {"index": 0}

    def test_the_turn_is_part_of_the_identity(self, tmp_path) -> None:
        # Same tool, same call number, different turn: turn 2's outcome
        # must not close turn 1's attempt. Dropping the turn from the key
        # once left the whole suite green, which is why this exists.
        log = AuditLog(path=tmp_path / "audit.jsonl")
        log.attempt("set_volume", {"percent": 10}, "yellow", 1)
        later = log.attempt("set_volume", {"percent": 20}, "yellow", 2)
        assert later["call"] == 1  # numbering restarts each turn
        log.succeeded("set_volume", 2, call=later["call"])
        assert [e["turn"] for e in log.unfinished()] == [1]

    def test_the_call_number_is_on_both_lines(self, tmp_path) -> None:
        # The pair of lines IS the record, so `tail` must be able to
        # pair them without any of this code: the outcome echoes the
        # attempt's number.
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path=path)
        log.attempt("set_volume", {"percent": 10}, "yellow", 1)
        second = log.attempt("set_volume", {"percent": 20}, "yellow", 1)
        log.succeeded("set_volume", 1, call=second["call"])
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [line["call"] for line in lines] == [1, 2, 2]

    def test_it_is_append_only_jsonl(self, tmp_path) -> None:
        # Readable with `tail` at three in the morning without any of
        # this code working.
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path=path)
        first = log.attempt("a", {}, "green", 1)
        log.succeeded("a", 1, call=first["call"])
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["event"] for line in lines)

    def test_a_failing_disk_does_not_stop_a_tool(self, tmp_path) -> None:
        # A full disk must not become an outage.
        log = AuditLog(path=tmp_path / "nonexistent" / "\x00bad" / "audit.jsonl")
        record = log.attempt("a", {}, "green", 1)
        assert record["event"] == ATTEMPTED
        assert len(log.entries) == 1

    def test_a_truncated_final_line_is_skipped_not_fatal(self, tmp_path) -> None:
        # A truncated last line is what a crash looks like, and that is
        # exactly when the log needs reading.
        path = tmp_path / "audit.jsonl"
        path.write_text('{"event":"attempted","tool":"a","turn":1}\n{"event":"succ')
        assert len(AuditLog.read(path)) == 1

    def test_a_missing_log_reads_as_empty(self, tmp_path) -> None:
        assert AuditLog.read(tmp_path / "nope.jsonl") == []

    def test_results_are_truncated(self, tmp_path) -> None:
        log = AuditLog(path=tmp_path / "audit.jsonl")
        first = log.attempt("a", {}, "green", 1)
        record = log.succeeded("a", 1, "x" * 5000, call=first["call"])
        assert len(record["result"]) <= 200

    def test_arguments_are_recorded_because_they_cannot_be_free_text(self, tmp_path) -> None:
        # Enums and ints by construction, so there is nothing to leak —
        # unlike the transcript, which is deliberately absent.
        log = AuditLog(path=tmp_path / "audit.jsonl")
        record = log.attempt("set_volume", {"percent": 30}, "yellow", 1)
        assert record["args"] == {"percent": 30}
        assert "transcript" not in record
