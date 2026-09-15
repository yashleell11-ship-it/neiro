"""Opening applications — the one tool whose blast radius crosses a
machine boundary, and the first tool in this registry that is a launch
rather than a query or a nudge to something already running.

Two walls keep this from being what egress.py exists to prevent
Hyprland's `exec` dispatcher from being (see egress.ALLOWED_DISPATCHERS
and its FORBIDDEN list — "exec"/"spawn" are banned there on purpose,
because that dispatcher shells out through Hyprland's Lua eval):

1. This never touches Hyprland's socket at all. Local launches go
   through `subprocess.Popen` with a literal argv list, `shell=False`
   always — there is no string for a shell or a Lua interpreter to
   parse. Remote launches go through a fixed PowerShell command built
   entirely from Elizabeth's own table, over the same SSH key already used
   for this machine's training operations.
2. `app` is a `Literal` the registry resolves against a table defined
   here, in Elizabeth's own code (see registry.py's `_field_is_safe`). The
   model can only ever pick a name from the table; it can never supply
   a path, an argv, or a command string.

Deliberately NOT in config.py — see egress.ALLOWED_DISPATCHERS for the
same rationale: an allowlist of what may run is a widenable one, and
widening it should be a code change with a test, not a settings edit.

**"pc" reaches the 3090 Ti box over Tailscale, not the open internet.**
See docs/DECISIONS.md (2026-09-15, "open_app crosses a machine
boundary") for why this doesn't violate Locality.LOCAL_PINNED the way
it looks like it should: Tier already models the box as two links to
one machine Yash owns (state.py), never an arbitrary host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

App = Literal["browser", "terminal", "discord", "vscode", "files"]

# Laptop (Linux/Hyprland). argv, never a shell string — verified
# present on this machine 2026-09-15 (`command -v`).
_LOCAL_APPS: dict[App, list[str]] = {
    "browser": ["firefox"],
    "terminal": ["kitty"],
    "discord": ["discord"],
    "vscode": ["code"],
    "files": ["dolphin"],
}

# Box (Windows). Full paths where App Paths / PATH resolution is not
# guaranteed for a GUI app launched non-interactively over SSH —
# verified present on the box 2026-09-15 (registry / Test-Path).
_BOX_COMMANDS: dict[App, str] = {
    "browser": '"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"',
    "terminal": "wt",
    "discord": '"C:\\Users\\yashl\\AppData\\Local\\Discord\\Update.exe" --processStart Discord.exe',
    "vscode": "code",
    "files": "explorer",
}

BOX_HOST = "yashl@100.86.69.33"  # Tailscale — see dashboard.py's note on why this over LAN
BOX_KEY = str(Path.home() / ".ssh" / "elizabeth_box_ed25519")
SSH_EXE = "ssh"
SSH_TIMEOUT_S = 15.0


class AppLaunchError(RuntimeError):
    """The app didn't start. The message is spoken to Yash."""


def open_local(app: App) -> str:
    argv = _LOCAL_APPS.get(app)
    if argv is None:
        raise AppLaunchError(f"I don't know how to open {app!r} here.")
    try:
        # start_new_session so the app outlives Elizabeth's own process tree,
        # same as any launcher — closing Elizabeth should not close Discord.
        subprocess.Popen(argv, start_new_session=True)
    except FileNotFoundError as exc:
        raise AppLaunchError(f"{app} isn't installed here.") from exc
    return f"Opening {app}."


def open_on_box(app: App) -> str:
    command = _BOX_COMMANDS.get(app)
    if command is None:
        raise AppLaunchError(f"I don't know how to open {app!r} on the PC.")
    try:
        result = subprocess.run(
            [
                SSH_EXE,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-i",
                BOX_KEY,
                BOX_HOST,
                f"powershell -NoProfile -Command Start-Process {command}",
            ],
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AppLaunchError("Couldn't reach the PC in time to open that.") from exc
    if result.returncode != 0:
        raise AppLaunchError(f"Couldn't reach the PC to open {app}.")
    return f"Opening {app} on the PC."


def open_app(target: Literal["here", "pc"], app: App) -> str:
    return open_on_box(app) if target == "pc" else open_local(app)
