"""What Elizabeth says when something goes wrong.

The gap this fills: the plan's own review found that every candidate
design had excellent *developer*-facing failure output (doctor rows,
loud assertions, readable tracebacks) and nothing at all for the user.
In a voice-first interface with no screen to fall back on, error speech
IS most of the perceived quality. The difference between "she's broken"
and "she didn't catch that" is entirely in whether she says something.

Design rules, each learned from a specific failure mode:

  - **Short.** 2-5 words. A long apology for a misheard word is worse
    than the misheard word.
  - **In character, not diagnostic.** "Say that again?" not
    "STT confidence below threshold". The log is for debugging; she
    isn't.
  - **Never blame the user.** "I didn't catch that" not "you were too
    quiet".
  - **Varied.** The same phrase every time turns a small annoyance into
    an irritating tic within a day. Rotate deterministically rather
    than randomly so a given failure sequence is reproducible in tests.
  - **Pre-synthesisable.** These are a fixed set, so they can be
    rendered to audio once at first run and played instantly, costing
    nothing at the moment a failure actually happens — which is
    precisely when there is no time to spare.
"""

from __future__ import annotations

from enum import StrEnum


class Failure(StrEnum):
    """Every way a turn can fail that the user should hear about."""

    NOT_HEARD = "not_heard"  # empty/rejected transcript
    MIC_MUTED = "mic_muted"  # capture device is muted
    LLM_DOWN = "llm_down"  # model server unreachable
    LLM_SLOW = "llm_slow"  # timed out mid-turn
    TOOL_FAILED = "tool_failed"  # an action didn't work
    TOOL_DENIED = "tool_denied"  # confirmation refused or timed out
    TIER_LOST = "tier_lost"  # dropped from the GPU box to laptop mid-session


# Multiple phrasings per failure so she doesn't develop a tic. Order is
# fixed, not shuffled — a given sequence of failures produces the same
# sequence of lines, which keeps tests meaningful and makes a recorded
# demo reproducible.
_LINES: dict[Failure, tuple[str, ...]] = {
    Failure.NOT_HEARD: (
        "Sorry, say that again?",
        "Didn't catch that.",
        "One more time?",
    ),
    Failure.MIC_MUTED: (
        "Your mic's muted.",
        "I can't hear anything — mic's off.",
    ),
    Failure.LLM_DOWN: (
        "My brain's not running.",
        "Can't reach the model.",
    ),
    Failure.LLM_SLOW: (
        "That took too long. Try again?",
        "Timed out, sorry.",
    ),
    Failure.TOOL_FAILED: (
        "That didn't work.",
        "Couldn't do it, sorry.",
    ),
    Failure.TOOL_DENIED: (
        "Okay, leaving it.",
        "Skipping that, then.",
    ),
    Failure.TIER_LOST: (
        "Lost the big model — still here though.",
        "Back on the laptop now.",
    ),
}


class ErrorSpeech:
    """Picks what she says, rotating through phrasings per failure kind.

    Stateful on purpose: the rotation is what stops her repeating
    herself, and it's per-session so a fresh run starts predictably at
    the first phrasing.
    """

    def __init__(self) -> None:
        self._counts: dict[Failure, int] = {}

    def line_for(self, failure: Failure) -> str:
        options = _LINES[failure]
        index = self._counts.get(failure, 0)
        self._counts[failure] = index + 1
        return options[index % len(options)]

    def reset(self) -> None:
        self._counts.clear()


def all_lines() -> list[str]:
    """Every line, for pre-synthesising them to audio at first run."""
    return [line for options in _LINES.values() for line in options]
