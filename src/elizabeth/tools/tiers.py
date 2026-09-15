"""Permission tiers, enforced in code rather than in the prompt.

A prompt is a suggestion. A local 4B model will eventually ignore one —
not maliciously, just by being a 4B model. So the tier of an action is a
property of the registry entry, checked before execution, and not
something the model is asked to respect.
"""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    GREEN = "green"
    """Read-only, no side effects. Runs immediately, no confirmation.

    System stats, window/workspace queries, what's playing, reading a
    file *listing*. Safe because there is nothing to undo.
    """

    YELLOW = "yellow"
    """Reversible side effects. Spoken confirmation AND a clickable
    notification, in parallel — the click always wins, because ASR
    hallucinates and a mouse doesn't.

    Volume, brightness, media control, focusing a window, launching an
    app from a fixed enum.
    """

    RED = "red"
    """Never reachable by voice at any confidence. Not merely gated —
    these are not registered as tools at all, so the model cannot even
    express them, and the egress filter refuses any dispatcher name
    outside the few the tools generate, as a second wall.

    Deleting or overwriting anything, sudo, `hyprctl eval/repl/plugin`,
    any shell, killing windows or processes, clipboard reads (a one-call
    credential exfiltration primitive), input synthesis, git push,
    sending anything off-machine the user didn't name.
    """
