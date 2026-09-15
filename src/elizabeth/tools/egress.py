"""The last wall before anything reaches Hyprland's socket.

**Why this module exists, and why it is not paranoia.**

On this machine (Hyprland 0.56.2, Lua config API), `hyprctl dispatch X`
wraps X as `return hl.dispatch(X)` and compiles it as Lua 5.5 — with
`os.execute` in scope. That was proven by experiment during the planning
research, not inferred: an injected payload ran, and
`type(os.execute) == "function"` evaluated true from inside it.

The consequence is that the spec's original safety claim — "the worst
case for a misheard command is that it sets the volume wrong" — was
false as written. A `focus_window(app: str)` implemented the obvious way,
by formatting a model-supplied string into a dispatch payload, is a full
code-execution escape reachable by background noise that Whisper
confidently mistranscribes.

**Three walls, and the first two are free:**

1. The model never emits a string that reaches a command line. It emits
   an integer index into a list Elizabeth produced this turn, or a `Literal`
   enum member. (registry.py)
2. Elizabeth resolves that index to a concrete value from its own table, so
   the value's provenance is Elizabeth, not the model. (hyprland.py)
3. This filter validates the complete, final payload before it touches
   the socket: the dispatcher name against an allowlist of what Elizabeth's
   tools actually generate, the argument table against a shape grammar,
   and the whole string against a denylist of Lua escape hatches. A
   backstop for a bug in 1 or 2, not a substitute.

A payload reaching wall 3 and failing means something upstream is
broken, so rejection is loud: it raises rather than sanitising. Silently
"cleaning" a payload would hide the bug that produced it.
"""

from __future__ import annotations

import re

# Substrings that must never appear in a payload bound for the socket.
# These are the Lua escape hatches plus the obvious shell routes. Matched
# case-insensitively against the whole payload, values included, so the
# error names the route that was attempted rather than just "bad name".
FORBIDDEN = (
    "os.",  # os.execute, os.exit, os.remove...
    "io.",  # io.popen, io.open
    "eval",  # hyprctl eval — arbitrary Lua
    "repl",  # hyprctl repl — arbitrary Lua
    "plugin",  # hyprctl plugin load — loads native code into the compositor
    "exec",  # exec / exec_cmd / exec_raw — all spawn through sh -c
    "spawn",  # a process by any other name
    "exit",  # hl.dsp.exit ends the session; os.exit ends the compositor
    "load",  # load / loadstring / dofile
    "dofile",
    "require",
    "package",
    "debug.",
    "getfenv",
    "setfenv",
    "rawset",
    "metatable",
)

# The dispatchers Elizabeth's tools actually build. Exact, case-sensitive
# names (Lua is case-sensitive, and the tools emit lowercase).
#
# Why a denylist is not enough on its own: the denylist above knows the
# shell routes someone thought of. `killactive`, `forcekillactive` and
# whatever the next Hyprland release adds are RED-tier actions (tiers.py)
# that look exactly like legitimate traffic — `hl.dsp.killactive({})` is
# a perfectly well-formed payload. Shape cannot tell them apart; only the
# name can. So the name is pinned to what the code generates, and adding
# one is a code change that arrives with a test, never a setting.
# Deliberately NOT in config.py for that reason: a tunable allowlist is
# a widenable one.
ALLOWED_DISPATCHERS = frozenset({"focus"})

# What a legitimate payload is allowed to look like. Deliberately an
# allowlist: enumerating what's safe is tractable, enumerating every way
# to smuggle Lua is not.
#
# Permits: hl.dsp.<name>({key=value, ...}) with values that are integers,
# simple quoted strings of safe characters, or booleans. The grammar only
# captures <name>; check() judges it against ALLOWED_DISPATCHERS.
_SAFE_PAYLOAD = re.compile(
    r"""^hl\.dsp\.(?P<dispatcher>[a-z_.]+)\(   # hl.dsp.focus(
        \{                             # opening brace
        \s*
        (?:[a-z_]+\s*=\s*              # key =
           (?:\d+                      #   an integer
             |true|false               #   a boolean
             |"[A-Za-z0-9:_\-. ]*"     #   a quoted string, safe chars only
           )
           \s*,?\s*
        )*
        \}
        \)$""",
    re.VERBOSE | re.IGNORECASE,
)


class EgressRejected(RuntimeError):
    """A payload failed validation. Loud on purpose — reaching this means
    a bug in the registry or the resolver, not a routine denial.
    """


def check(payload: str) -> str:
    """Validate a complete Lua payload. Returns it unchanged, or raises.

    Never sanitises. A payload that needs cleaning is a payload built
    wrong, and quietly fixing it would hide that.
    """
    if not payload or not payload.strip():
        raise EgressRejected("empty payload")

    lowered = payload.lower()
    for banned in FORBIDDEN:
        if banned in lowered:
            raise EgressRejected(
                f"payload contains forbidden substring {banned!r} — this should "
                f"be impossible if the registry only accepts typed indices and "
                f"enums, so treat it as a bug upstream, not a denied request. "
                f"Payload: {payload!r}"
            )

    match = _SAFE_PAYLOAD.match(payload.strip())
    if not match:
        raise EgressRejected(
            f"payload does not match the allowed shape "
            f"hl.dsp.<name>({{key=value, ...}}) with integer, boolean, or "
            f"simple-quoted-string values. Payload: {payload!r}"
        )

    dispatcher = match.group("dispatcher")
    if dispatcher not in ALLOWED_DISPATCHERS:
        raise EgressRejected(
            f"dispatcher {dispatcher!r} is not one Elizabeth's tools generate "
            f"(allowed: {sorted(ALLOWED_DISPATCHERS)}). A well-formed payload "
            f"naming an unlisted dispatcher is what a RED-tier action looks "
            f"like from here. Payload: {payload!r}"
        )

    return payload
