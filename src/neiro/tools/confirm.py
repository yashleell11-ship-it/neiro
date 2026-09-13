"""Asking before doing a YELLOW thing, in two channels at once.

**The click always wins.** Speech recognition hallucinates — the
confidence floor in `stt/faster_whisper.py` exists because Whisper
confidently transcribes silence as "thank you". A mouse does not
hallucinate. So a notification with Yes/No buttons is raised in parallel
with the spoken question, and whichever answers first decides.

**The model never participates.** The nonce is computed in the registry
and never reaches the LLM, and the answer is matched against a *closed
lexicon* here rather than parsed by anything. A confirmation an LLM can
influence is not a confirmation.

**Ambiguity and timeout are DENY.** Both. An assistant that acts on "uh"
is worse than one that asks twice.

The lexicon includes Hindi and Hinglish because that is how Yash
actually answers — "haan", "haan ji", "theek hai", "nahi", "rehne do" —
and requiring English for the safety mechanism specifically would be the
wrong place to demand it.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from neiro.config import SttConfig

# Closed lexicons. Anything not in either is ambiguous, and ambiguous is
# no. Kept small deliberately: every word added is a word that can be
# hallucinated into a yes.
# NOTE the absence of "okay" and "ok". They are the most natural way to
# agree in English AND two of Whisper's commonest silence
# hallucinations — an empty room transcribes as "Okay." often enough
# that a test caught it here: "okay." normalises to "okay", which
# matched, so hallucinated silence would have authorised a YELLOW
# action. The confidence floor catches most of those, but a safety
# mechanism must not rest on one check. There are plenty of other
# natural ways to say yes, and none of them are what silence sounds
# like. ("ha" is gone for the same reason — it is a laugh.)
YES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "go",
        "go ahead",
        "do it",
        "please",
        "please do",
        "confirm",
        "affirmative",
        "haan",
        "haan ji",
        "haanji",
        "theek hai",
        "thik hai",
        "kar do",
        "karo",
    }
)
NO = frozenset(
    {
        "no",
        "nope",
        "nah",
        "don't",
        "dont",
        "stop",
        "cancel",
        "never mind",
        "nevermind",
        "forget it",
        "wait",
        "hold on",
        "negative",
        "nahi",
        "nahin",
        "mat karo",
        "rehne do",
        "ruko",
        "nahi ji",
    }
)

# Whisper's silence hallucinations. These are what an empty room
# transcribes as, and none of them may ever mean yes.
# Stored already NORMALISED, because the check runs against normalised
# text — keeping "okay." here while comparing against "okay" is exactly
# how one slipped through.
HALLUCINATIONS = frozenset(
    {"thank you", "thanks for watching", "you", "bye", "okay", "ok", "uh", "um", "ha"}
)

MAX_WORDS = 4  # a confirmation is short; a sentence is a different answer
WINDOW_SECONDS = 6.0


@dataclass(frozen=True)
class Answer:
    decision: str  # "yes" | "no" | "ambiguous" | "timeout"
    source: str  # "voice" | "click" | "timeout"
    heard: str = ""

    @property
    def allowed(self) -> bool:
        # Only an explicit yes. Everything else — including ambiguity and
        # silence — is no.
        return self.decision == "yes"


def normalise(text: str) -> str:
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace() or c == "'")
    return " ".join(cleaned.split())


def interpret(
    transcript: str,
    no_speech_prob: float = 0.0,
    avg_logprob: float = 0.0,
    *,
    stt: SttConfig | None = None,
) -> Answer:
    """Turn what was heard into a decision. Never asks a model anything.

    The STT confidence floors are applied here too, not only upstream:
    this is the one place where being wrong changes the machine, so it
    re-checks rather than trusting that someone else did.

    Re-checked against the SAME floors, though — `stt` is the loaded
    `cfg.stt`, not a copy of its numbers. This function once carried its
    own `0.4` and `-1.0`, so tightening the floor in config after a run
    of hallucinated confirmations would have tightened transcription
    and left the one check that gates side effects exactly where it
    was. Whoever wires the voice confirmer must pass `cfg.stt`: the
    default below is the config's default, which drifts from
    `~/.config/neiro/config.toml` the moment it is edited.
    """
    floors = stt if stt is not None else SttConfig()
    text = normalise(transcript)
    if not text:
        return Answer("ambiguous", "voice", transcript)
    if text in HALLUCINATIONS:
        return Answer("ambiguous", "voice", transcript)
    if no_speech_prob > floors.no_speech_prob_floor or avg_logprob < floors.avg_logprob_floor:
        return Answer("ambiguous", "voice", transcript)
    if len(text.split()) > MAX_WORDS:
        # "yes I think we should probably do that" is a sentence, not a
        # confirmation, and matching a substring of it is how "no, don't"
        # becomes a yes.
        return Answer("ambiguous", "voice", transcript)
    if text in YES:
        return Answer("yes", "voice", transcript)
    if text in NO:
        return Answer("no", "voice", transcript)
    return Answer("ambiguous", "voice", transcript)


@dataclass
class NotificationConfirmer:
    """The clickable half. `notify-send -A` blocks and prints the chosen
    action id, which is the whole mechanism.
    """

    timeout_s: float = WINDOW_SECONDS
    runner: object = field(default=None)

    def ask(self, question: str) -> Answer:
        cmd = [
            "notify-send",
            "--urgency=critical",
            f"--expire-time={int(self.timeout_s * 1000)}",
            "--action=yes=Yes",
            "--action=no=No",
            "--wait",
            "Neiro",
            question,
        ]
        try:
            run = self.runner or subprocess.run
            result = run(cmd, capture_output=True, text=True, timeout=self.timeout_s + 1)
            chosen = (result.stdout or "").strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
            return Answer("timeout", "timeout")
        if chosen == "yes":
            return Answer("yes", "click")
        if chosen == "no":
            return Answer("no", "click")
        return Answer("timeout", "timeout")


def decide(voice: Answer | None, click: Answer | None) -> Answer:
    """Combine the two channels. The click wins when both answered.

    Not "the first to answer": if a click exists it is authoritative,
    because it is the channel that cannot hallucinate. A voice yes and a
    click no must resolve to no.
    """
    if click is not None and click.decision in ("yes", "no"):
        return click
    if voice is not None and voice.decision in ("yes", "no"):
        return voice
    return Answer("timeout", "timeout")
