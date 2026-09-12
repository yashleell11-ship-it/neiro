"""Talk to Hyprland over its raw UNIX socket.

Two rules, both learned the hard way in the planning research:

**Always connect -> send -> read to EOF -> close.** Hyprland evaluates
the request socket synchronously, and its own wiki states that an
unclosed connection freezes the compositor until a five-second timeout.
A pooled connection, or a crash between send and close, takes the user's
whole desktop down with it — the single most likely way this project
makes the laptop feel broken. So: no connection pooling, no keep-alive,
a `with` block every time, and an explicit socket timeout.

**Never use the `hyprctl` subprocess for reads.** Measured during
research: 0.021 ms median over the socket versus 3 ms through the
subprocess — about 150x, on something the always-listening loop will do
constantly.

Writes are a different matter entirely. `hyprctl dispatch X` on this
machine compiles X as Lua 5.5 with `os.execute` in scope, so every write
payload is built from Neiro's own resolved values and validated by
egress.check() before it touches the socket. See egress.py.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from neiro.tools.egress import check as egress_check

SOCKET_TIMEOUT_S = 0.5


class HyprlandUnavailable(RuntimeError):
    """Not running under Hyprland, or the socket has gone."""


def socket_path() -> Path:
    """Path to the request socket for the running Hyprland instance."""
    signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if not signature:
        raise HyprlandUnavailable(
            "HYPRLAND_INSTANCE_SIGNATURE is not set — not running under Hyprland. "
            "Neiro's window tools are Hyprland-specific; the voice loop works "
            "without them."
        )
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "hypr" / signature / ".socket.sock"


def request(command: str, timeout_s: float = SOCKET_TIMEOUT_S) -> str:
    """Send one command, return the full response, always closing.

    `command` is a read command like ``j/clients``. Writes go through
    `dispatch()`, which validates first.
    """
    path = socket_path()
    if not path.exists():
        raise HyprlandUnavailable(f"Hyprland socket not found at {path}")

    # A `with` block so the socket closes even if read/send raises —
    # leaving it open is what freezes the desktop.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(str(path))
        sock.sendall(command.encode())
        chunks = []
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode(errors="replace")


def request_json(command: str) -> list | dict:
    """Send a `j/...` command and parse the JSON response."""
    raw = request(command)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HyprlandUnavailable(
            f"Hyprland returned non-JSON for {command!r}: {raw[:200]!r}"
        ) from exc


@dataclass(frozen=True)
class Window:
    """One open window, as Neiro sees it.

    `index` is the ONLY handle the model is ever given. It indexes into
    the list Neiro produced this turn; Neiro resolves it back to
    `address` itself. The model never sees or supplies an address, a
    class, or a title that reaches a command line.
    """

    index: int
    address: str
    app: str
    title: str
    workspace: int

    def describe(self) -> str:
        """What Neiro says about this window out loud."""
        return f"{self.app}: {self.title}" if self.title else self.app


def list_windows() -> list[Window]:
    """Current windows, indexed. GREEN tier — read-only."""
    raw = request_json("j/clients")
    windows = []
    for i, client in enumerate(raw):
        workspace = client.get("workspace") or {}
        windows.append(
            Window(
                index=i,
                address=client.get("address", ""),
                app=client.get("class", "") or "unknown",
                title=client.get("title", ""),
                workspace=workspace.get("id", 0),
            )
        )
    return windows


def active_window() -> Window | None:
    """The focused window, or None if nothing is focused. GREEN tier."""
    raw = request_json("j/activewindow")
    if not raw or not raw.get("address"):
        return None
    workspace = raw.get("workspace") or {}
    return Window(
        index=-1,  # not from an indexed list
        address=raw.get("address", ""),
        app=raw.get("class", "") or "unknown",
        title=raw.get("title", ""),
        workspace=workspace.get("id", 0),
    )


def dispatch(payload: str) -> str:
    """Send a validated write payload.

    `payload` must already be a complete Lua dispatch expression built
    by Neiro from its own resolved values — never interpolated from
    anything the model supplied directly. egress.check() is the backstop,
    not the primary defence.
    """
    egress_check(payload)
    return request(f"dispatch {payload}")


def focus_window_by_index(index: int, windows: list[Window]) -> str:
    """YELLOW tier. Resolve an index Neiro itself produced into an
    address, then dispatch.

    The model supplies only `index`. If it supplies something out of
    range that's a plain IndexError on our side, not a command with a
    hostile string in it.
    """
    if not 0 <= index < len(windows):
        raise IndexError(f"no window with index {index} (have {len(windows)})")
    address = windows[index].address
    if not address.startswith("0x"):
        raise ValueError(f"refusing to dispatch a suspicious address: {address!r}")
    return dispatch(f'hl.dsp.focus({{window="address:{address}"}})')


def switch_workspace(number: int) -> str:
    """YELLOW tier. `number` is an int from a constrained range, so
    there is no string to escape in the first place.
    """
    if not 1 <= number <= 10:
        raise ValueError(f"workspace must be 1-10, got {number}")
    return dispatch(f"hl.dsp.focus({{workspace={number}}})")
