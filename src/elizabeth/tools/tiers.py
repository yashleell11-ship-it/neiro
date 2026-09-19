"""Permission tiers, enforced in code rather than in the prompt.

A prompt is a suggestion. A local 4B model will eventually ignore one —
not maliciously, just by being a 4B model. So the tier of an action is a
property of the registry entry, checked before execution, and not
something the model is asked to respect.
"""

from __future__ import annotations

from enum import StrEnum

from elizabeth.state import Tier as StateTier


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


class Reach(StrEnum):
    """How far a tool's side effect travels.

    Separate from `Tier`, which asks "how bad is this if it goes wrong?".
    This asks "whose network does it cross?" — and the two are genuinely
    independent: `open_app` is YELLOW whether it opens Firefox here or
    Edge on the box, but only one of those leaves the machine.

    It exists because docs/PRIVACY.md makes a claim this code has to be
    able to keep. `open_app(target="pc")` and `web_search` were added on
    Yash's direct request and deliberately break "fully local"; what was
    left undone, and flagged rather than skipped, was that neither
    consulted the turn's NETWORK tier (state.Tier: LOCAL/LAN/TUNNEL). So
    nothing stopped the model reaching the box again, or the open
    internet, on a turn that was *already* routing through a tunnel —
    the compounding-exposure question PRIVACY.md exists to be honest
    about.

    The policy is deliberately asymmetric: it can only ever refuse more
    than before, never allow more. At home (LOCAL, or LAN over his own
    ethernet) both tools behave exactly as they did. Over the hostel
    TUNNEL, anything that leaves the machine is refused — issuing
    desktop commands to the house, or putting a spoken query onto the
    open internet through an extra hop, is a materially different act
    from doing it while sitting in front of the machine.
    """

    LOCAL = "local"
    """Never leaves the machine Yash is sitting at. Every tool but two."""

    OWN_MACHINES = "own_machines"
    """Reaches his other machine — today only `open_app(target="pc")`."""

    INTERNET = "internet"
    """Crosses onto the open internet — today only `web_search`."""

    def allows(self, tier: "StateTier") -> bool:
        """May a tool with this reach run on `tier`?

        Mirrors `state.Locality.allows()` on purpose: the rule lives in
        code the registry calls, not in a comment someone reads once.
        """
        if self is Reach.LOCAL:
            return True
        return tier is not StateTier.TUNNEL


class Source(StrEnum):
    """Which surface asked for a tool call.

    A third axis, and independent of the other two for the same reason
    `Reach` is independent of `Tier`: `Tier` asks how bad this is if it
    goes wrong, `Reach` asks whose network it crosses, and `Source` asks
    **who is allowed to ask**. A tool can be cheap, local and still be
    something a camera must never be able to trigger.

    It exists because v2 puts a second caller behind the registry. Until
    now every call came from the LLM, having been through a system
    prompt, a tool schema and (for YELLOW) a confirmation. A gesture has
    been through none of those — it is a hand moving in a room, and the
    recogniser cannot tell a deliberate swipe from an identical
    accidental one. So gesture-eligibility is opt-in per tool
    (`ToolSpec.gesture_ok`), defaulting to False, exactly as `reach`
    defaults to LOCAL: the narrow value is the safe one, and a tool that
    wants the wider surface has to say so in its own definition.
    """

    VOICE = "voice"
    """The LLM asked, on the user's spoken turn. The historical default."""

    GESTURE = "gesture"
    """v2's camera saw a gesture. No language model was involved."""
