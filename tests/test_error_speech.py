"""Tests for Neiro's error speech.

Mostly guarding the product rules rather than the logic: short, varied,
never blaming the user, and complete (every failure kind has something
to say, so no code path can end in silence).
"""

from __future__ import annotations

from neiro.speech.errors import ErrorSpeech, Failure, all_lines


class TestCoverage:
    def test_every_failure_kind_has_lines(self) -> None:
        # A failure with no line means a silent failure, which is the
        # exact thing this module exists to prevent.
        speech = ErrorSpeech()
        for failure in Failure:
            assert speech.line_for(failure).strip()


class TestRotation:
    def test_repeats_are_varied(self) -> None:
        # The same phrase every time becomes an irritating tic fast.
        speech = ErrorSpeech()
        first = speech.line_for(Failure.NOT_HEARD)
        second = speech.line_for(Failure.NOT_HEARD)
        assert first != second

    def test_rotation_wraps_around(self) -> None:
        speech = ErrorSpeech()
        seen = [speech.line_for(Failure.NOT_HEARD) for _ in range(6)]
        assert seen[0] == seen[3]  # 3 phrasings, wraps on the 4th

    def test_rotation_is_per_failure_kind(self) -> None:
        # Hearing "didn't catch that" three times shouldn't advance the
        # rotation for an unrelated failure — LLM_DOWN should still be
        # on its own first phrasing.
        speech = ErrorSpeech()
        for _ in range(3):
            speech.line_for(Failure.NOT_HEARD)

        fresh = ErrorSpeech()
        assert speech.line_for(Failure.LLM_DOWN) == fresh.line_for(Failure.LLM_DOWN)

    def test_deterministic_across_instances(self) -> None:
        # Not random: a given failure sequence produces the same lines,
        # so a recorded demo is reproducible and tests stay meaningful.
        a, b = ErrorSpeech(), ErrorSpeech()
        assert [a.line_for(Failure.NOT_HEARD) for _ in range(4)] == [
            b.line_for(Failure.NOT_HEARD) for _ in range(4)
        ]

    def test_reset_returns_to_the_first_phrasing(self) -> None:
        speech = ErrorSpeech()
        first = speech.line_for(Failure.NOT_HEARD)
        speech.line_for(Failure.NOT_HEARD)
        speech.reset()
        assert speech.line_for(Failure.NOT_HEARD) == first


class TestProductRules:
    def test_all_lines_are_short(self) -> None:
        # 2-5 words. A long apology is worse than the original problem.
        for line in all_lines():
            assert len(line.split()) <= 8, f"too wordy: {line!r}"

    def test_no_line_blames_the_user(self) -> None:
        banned = ("you were", "you didn't", "your fault", "too quiet", "speak up")
        for line in all_lines():
            lowered = line.lower()
            for phrase in banned:
                assert phrase not in lowered, f"{line!r} blames the user"

    def test_no_line_leaks_diagnostics(self) -> None:
        # She isn't the log. "confidence below threshold" is for the
        # JSONL, not for speech.
        banned = ("threshold", "confidence", "exception", "error code", "stderr", "traceback")
        for line in all_lines():
            lowered = line.lower()
            for phrase in banned:
                assert phrase not in lowered, f"{line!r} reads like a log line"

    def test_lines_are_speakable_not_markup(self) -> None:
        for line in all_lines():
            assert "[" not in line and "<" not in line, f"{line!r} contains markup"

    def test_all_lines_is_complete(self) -> None:
        assert len(all_lines()) >= len(list(Failure))
