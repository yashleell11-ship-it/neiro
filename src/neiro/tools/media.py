"""Media control over MPRIS, via playerctl.

GREEN for reading what's playing, YELLOW for changing it. No new Python
dependency: playerctl is already installed and speaks MPRIS to whatever
is running (Spotify, a browser tab, mpv).

Every action here is a `Literal` enum member, never free text — there is
no string from the model that reaches a command line.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

ACTIONS = ("play", "pause", "play-pause", "next", "previous")

# playerctl says this on stderr and exits non-zero when nothing is
# running. It's the normal case, not an error worth surfacing.
_NO_PLAYERS = "no players found"


def _run(args: list[str], timeout: float = 2.0) -> tuple[str, bool]:
    """Returns (stdout, ok)."""
    try:
        result = subprocess.run(
            ["playerctl", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "", False
    if result.returncode != 0:
        return "", False
    return result.stdout.strip(), True


@dataclass(frozen=True)
class NowPlaying:
    artist: str
    title: str
    status: str  # "Playing" | "Paused" | "Stopped"

    def describe(self) -> str:
        """What she says out loud."""
        if self.artist and self.title:
            body = f"{self.title} by {self.artist}"
        else:
            body = self.title or self.artist or "something"
        if self.status.lower() == "paused":
            return f"{body}, paused"
        return body


def get_now_playing() -> NowPlaying | None:
    """GREEN tier. None when nothing is playing — which is normal, not
    a failure.
    """
    out, ok = _run(["-f", "{{status}}\t{{artist}}\t{{title}}", "metadata"])
    if not ok or not out:
        return None
    parts = out.split("\t")
    while len(parts) < 3:
        parts.append("")
    status, artist, title = parts[0], parts[1], parts[2]
    if not (artist or title):
        return None
    return NowPlaying(artist=artist, title=title, status=status)


def media_control(action: str) -> str:
    """YELLOW tier. `action` is a Literal enum member.

    Deliberately excludes `open` (which takes a URI — a string that
    reaches a command line) and `position` (which is fiddly by voice and
    easy to get destructively wrong).
    """
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    _out, ok = _run([action])
    if not ok:
        return "nothing's playing"
    return action.replace("-", " ")
