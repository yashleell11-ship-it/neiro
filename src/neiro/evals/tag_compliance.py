"""Does she actually open every reply with a well-formed emotion tag?

The `<e:LABEL:D>` tag is the single channel carrying her state to her
voice, her face, and the next turn's prompt. If it is missing the parser
falls back to neutral after 32 characters — which is correct behaviour
and also means a *silent* degradation: she keeps talking, her face just
stops meaning anything. Stage 1's gate is ≥98% compliance, and this is
what measures it.

Three failures are counted separately because they need different fixes:

  - **missing** — no tag at all. A prompt problem.
  - **malformed** — a tag-shaped thing the parser rejects (`<e:happy>`,
    `<e:excited:7>`, `<e:happy:12>`). A prompt problem with a different
    cause: she knows about the tag but not its grammar.
  - **misplaced** — a valid tag somewhere other than the very start, or
    more than one. This is the dangerous one: the parser takes the first
    and the rest get *spoken aloud*.

Also reported, without a gate: the distribution of labels. The character
prompt warns that "a flat `neutral:5` on everything makes you a robot
with a face bolted on", and a model can score 100% compliance while
being exactly that. Compliance and expressiveness are different
properties and this refuses to conflate them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from neiro.state import EmotionLabel

# Anything that looks like an attempt at the tag, so a malformed one is
# distinguishable from no tag at all.
TAG_SHAPED = re.compile(r"<\s*e\s*:[^>]*>", re.IGNORECASE)
# The exact grammar the parser accepts.
VALID_TAG = re.compile(r"^<e:(happy|angry|sad|relaxed|surprised|neutral):([0-9])>")

VERDICTS = ("ok", "missing", "malformed", "misplaced")


@dataclass
class TagResult:
    reply: str
    verdict: str
    label: str | None = None
    intensity: int | None = None

    @property
    def compliant(self) -> bool:
        return self.verdict == "ok"


def check(reply: str) -> TagResult:
    """Classify one reply."""
    text = reply.lstrip()
    match = VALID_TAG.match(text)
    shaped = TAG_SHAPED.findall(text)

    if match is None:
        return TagResult(reply, "malformed" if shaped else "missing")
    # A second tag anywhere after the first is `misplaced`: the parser
    # keeps the first and everything else is spoken out loud.
    if len(shaped) > 1:
        return TagResult(reply, "misplaced", match.group(1), int(match.group(2)))
    return TagResult(reply, "ok", match.group(1), int(match.group(2)))


@dataclass
class ComplianceReport:
    results: list[TagResult] = field(default_factory=list)
    gate: float = 0.98

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def rate(self) -> float:
        return sum(r.compliant for r in self.results) / self.n if self.n else 0.0

    @property
    def passed(self) -> bool:
        return self.n > 0 and self.rate >= self.gate

    def by_verdict(self) -> dict[str, int]:
        counts = Counter(r.verdict for r in self.results)
        return {v: counts.get(v, 0) for v in VERDICTS}

    def by_label(self) -> dict[str, int]:
        counts = Counter(r.label for r in self.results if r.label)
        return dict(counts.most_common())

    @property
    def flat(self) -> bool:
        """True if one label accounts for almost everything.

        Not part of the gate — a model can be 100% compliant and still be
        "a robot with a face bolted on". Reported so the two properties
        are never conflated.
        """
        labels = self.by_label()
        total = sum(labels.values())
        return bool(total) and max(labels.values()) / total > 0.8

    def summary(self) -> dict:
        return {
            "n": self.n,
            "rate": round(self.rate, 4),
            "gate": self.gate,
            "passed": self.passed,
            "by_verdict": self.by_verdict(),
            "by_label": self.by_label(),
            "unused_labels": sorted({e.value for e in EmotionLabel} - set(self.by_label())),
            "flat": self.flat,
        }


def score(replies: list[str], gate: float = 0.98) -> ComplianceReport:
    return ComplianceReport([check(r) for r in replies], gate=gate)
