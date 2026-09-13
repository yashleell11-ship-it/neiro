"""An append-only record of everything Neiro did, or tried to do.

Written **before and after** execution, deliberately. A log written only
after success cannot answer the one question that matters when something
went wrong: *did it run?* A tool that crashed the daemon halfway, or hung,
or changed something and then failed, leaves only the "attempt" line —
and that gap is the evidence.

Append-only and plain JSONL, so it can be read with `tail` at three in
the morning without any of this code working.

**Nothing here records what was said.** The transcript is not in the
audit log: this is a record of *actions*, and conversation content has a
different retention story (`docs/PRIVACY.md`). Arguments are recorded
because they are enums and integers by construction — there is no free
text to leak.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUDIT_PATH = Path.home() / ".local/state/neiro/audit.jsonl"

# Outcomes. "attempted" is never rewritten to something else — the pair
# of lines IS the record, and a lone "attempted" means it did not finish.
ATTEMPTED = "attempted"
SUCCEEDED = "succeeded"
FAILED = "failed"
DENIED = "denied"
RATE_LIMITED = "rate-limited"
OUTCOMES = frozenset({SUCCEEDED, FAILED, DENIED, RATE_LIMITED})


@dataclass
class AuditLog:
    path: Path = AUDIT_PATH
    clock: Any = time.time
    _entries: list[dict] = field(default_factory=list)
    # How many calls each turn has made so far, so an attempt can be
    # numbered within its turn.
    # Calls are numbered within the turn. Turns are sequential, so only
    # the current one needs remembering — a dict keyed by turn id grew by
    # one entry per turn for the life of the daemon.
    _numbering_turn: int | None = None
    _calls_this_turn: int = 0

    def _write(self, record: dict) -> dict:
        record = {"t": round(self.clock(), 3), **record}
        self._entries.append(record)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode plus a single write call: two daemons logging
            # at once interleave whole lines rather than corrupting one.
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, ValueError, TypeError):
            # A failing audit log must never stop a tool from running, or
            # a full disk becomes an outage. Deliberately broad: a bad
            # path raises ValueError rather than OSError, and the whole
            # point is that NO logging problem reaches the caller. The
            # in-memory copy survives either way.
            pass
        return record

    def attempt(self, tool: str, args: dict, tier: str, turn_id: int) -> dict:
        """Before execution. The existence of this line without a
        matching outcome is what tells you a tool did not finish.

        Numbered within the turn: `call` is the identity the outcome
        line must echo, because (tool, turn) is not one. The model can
        call the same tool twice in a turn, and one outcome must not
        close both.
        """
        if turn_id != self._numbering_turn:
            self._numbering_turn, self._calls_this_turn = turn_id, 0
        self._calls_this_turn += 1
        call = self._calls_this_turn
        return self._write(
            {
                "event": ATTEMPTED,
                "tool": tool,
                "args": args,
                "tier": tier,
                "turn": turn_id,
                "call": call,
            }
        )

    # `call` on every outcome is the number the attempt line was given.
    # Keyword-only and required: forgetting it is a TypeError at the
    # call site, not an outcome that quietly closes nothing.

    def succeeded(self, tool: str, turn_id: int, result: str = "", *, call: int) -> dict:
        # Truncated: a tool result is spoken aloud, so it is short by
        # design, but a future one returning a page of text should not
        # bloat the log.
        return self._write(
            {
                "event": SUCCEEDED,
                "tool": tool,
                "turn": turn_id,
                "call": call,
                "result": result[:200],
            }
        )

    def failed(self, tool: str, turn_id: int, error: str, *, call: int) -> dict:
        return self._write(
            {"event": FAILED, "tool": tool, "turn": turn_id, "call": call, "error": error[:200]}
        )

    def denied(self, tool: str, turn_id: int, reason: str, *, call: int) -> dict:
        return self._write(
            {"event": DENIED, "tool": tool, "turn": turn_id, "call": call, "reason": reason[:200]}
        )

    def rate_limited(self, tool: str, turn_id: int, *, call: int) -> dict:
        return self._write({"event": RATE_LIMITED, "tool": tool, "turn": turn_id, "call": call})

    # -- reading --------------------------------------------------------

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def unfinished(self) -> list[dict]:
        """Attempts with no matching outcome — the interesting ones.

        Matched on (tool, turn, call). Tool and turn alone are not an
        identity: the same tool called twice in one turn shared a key,
        so one success closed both attempts and the call that hung was
        exactly the one that vanished. The call number is stamped on the
        attempt and echoed by its outcome, so an outcome closes one
        attempt — its own. Pairing by count would get the number right
        and the arguments wrong when the second call is the one that
        finished, and the arguments are what you read at three in the
        morning.
        """
        closed = {
            (e["tool"], e["turn"], e["call"]) for e in self._entries if e["event"] in OUTCOMES
        }
        return [
            e
            for e in self._entries
            if e["event"] == ATTEMPTED and (e["tool"], e["turn"], e["call"]) not in closed
        ]

    @classmethod
    def read(cls, path: Path | None = None) -> list[dict]:
        """Every record on disk. A corrupt line is skipped, not fatal —
        a truncated final line is what a crash looks like, and that is
        exactly when the log needs reading.
        """
        path = path or AUDIT_PATH
        records = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return records
