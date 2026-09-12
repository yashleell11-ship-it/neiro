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
   an integer index into a list Neiro produced this turn, or a `Literal`
   enum member. (registry.py)
2. Neiro resolves that index to a concrete value from its own table, so
   the value's provenance is Neiro, not the model. (hyprland.py)
3. This filter validates the complete, final payload before it touches
   the socket — a backstop for a bug in 1 or 2, not a substitute.

A payload reaching wall 3 and failing means something upstream is
broken, so rejection is loud: it raises rather than sanitising. Silently
"cleaning" a payload would hide the bug that produced it.
"""

from __future__ import annotations

import re

# Substrings that must never appear in a payload bound for the socket.
# These are the Lua escape hatches plus the obvious shell routes. Matched
# case-insensitively against the whole payload.
FORBIDDEN = (
    "os.",  # os.execute, os.exit, os.remove...
    "io.",  # io.popen, io.open
    "eval",  # hyprctl eval — arbitrary Lua
    "repl",  # hyprctl repl — arbitrary Lua
    "plugin",  # hyprctl plugin load — loads native code into the compositor
    "exec_cmd",  # runs sh -c
    "exec_raw",
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

# What a legitimate payload is allowed to look like. Deliberately an
# allowlist: enumerating what's safe is tractable, enumerating every way
# to smuggle Lua is not.
#
# Permits: hl.dsp.<name>({key=value, ...}) with values that are integers,
# simple quoted strings of safe characters, or booleans.
_SAFE_PAYLOAD = re.compile(
    r"""^hl\.dsp\.[a-z_.]+\(          # hl.dsp.focus(  /  hl.dsp.window.move(
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

    if not _SAFE_PAYLOAD.match(payload.strip()):
        raise EgressRejected(
            f"payload does not match the allowed shape "
            f"hl.dsp.<name>({{key=value, ...}}) with integer, boolean, or "
            f"simple-quoted-string values. Payload: {payload!r}"
        )

    return payload
