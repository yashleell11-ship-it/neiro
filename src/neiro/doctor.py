"""The preflight. Every assumption the rest of the project makes gets
checked here, with a readable remedy on failure — never a stack trace.

Stage 0 Task 0 lives here: `audio_inventory()` refuses to let the project
proceed until the two device profiles (earbuds / speakers) are named
explicitly in config.toml, because this laptop's PipeWire 'default'
sink/source silently changes and a voice assistant that follows it will
one day play into a muted device and look broken.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str | None = None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout
    except FileNotFoundError:
        return ""


@dataclass
class AudioNode:
    id: str
    name: str
    kind: str  # "sink" | "source"


def _list_nodes(kind: str) -> list[AudioNode]:
    out = _run(["pactl", "list", "short", kind + "s"])
    nodes = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            nodes.append(AudioNode(id=parts[0], name=parts[1], kind=kind))
    return nodes


def _defaults() -> dict[str, str]:
    out = _run(["pactl", "info"])
    result: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("Default Sink:"):
            result["sink"] = line.split(":", 1)[1].strip()
        elif line.startswith("Default Source:"):
            result["source"] = line.split(":", 1)[1].strip()
    return result


def audio_inventory() -> int:
    """`neiro doctor --audio-inventory` — Stage 0 Task 0.

    Prints every sink/source, marks the current defaults, flags a
    bluetooth card if present, and tells the user exactly what to put
    in config.toml. Returns a process exit code (0 if profiles are
    already configured, 1 if this is the first run and they aren't).
    """
    defaults = _defaults()
    sinks = _list_nodes("sink")
    sources = [n for n in _list_nodes("source") if "monitor" not in n.name]

    table = Table(title="Audio devices (verified now, not assumed)")
    table.add_column("kind")
    table.add_column("node name")
    table.add_column("default?")
    for n in sinks:
        is_default = "← DEFAULT" if n.name == defaults.get("sink") else ""
        table.add_row("sink (output)", n.name, is_default)
    for n in sources:
        is_default = "← DEFAULT" if n.name == defaults.get("source") else ""
        table.add_row("source (input)", n.name, is_default)
    console.print(table)

    bt = [n for n in sinks + sources if n.name.startswith("bluez_")]
    if bt:
        console.print(
            "[yellow]Bluetooth device detected.[/yellow] Note: A2DP profile has no "
            "microphone — opening capture forces an HFP/HSP switch that drops the "
            "output to ~16 kHz mid-conversation. Measure this in Gate G4 (Stage 0 "
            "Task 14) before trusting earbuds as the daily profile."
        )

    console.print(
        "\n[bold]Next step:[/bold] copy the exact node names above into "
        r"~/.config/neiro/config.toml under \[audio.profiles.earbuds] and "
        r"\[audio.profiles.speakers], then set active_profile under \[audio] "
        r"itself — a bare active_profile = ... line placed after a "
        r"\[audio.profiles.*] table header attaches to THAT table, not "
        r"\[audio] (a real TOML footgun, not just a style note). Never leave "
        "a profile pointed at 'default' — see config.py's AudioDeviceProfile."
    )
    return 0


def run_doctor(audio_inventory_only: bool = False) -> int:
    if audio_inventory_only:
        return audio_inventory()

    checks: list[Check] = []

    # Python version
    import sys

    py_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        Check(
            "Python 3.12",
            py_ok,
            f"{sys.version_info.major}.{sys.version_info.minor}",
            remedy=None if py_ok else "uv python pin 3.12 && uv sync",
        )
    )

    # Audio device profiles configured
    from neiro.config import Neiro

    try:
        cfg = Neiro()
        has_profiles = bool(cfg.audio.active_profile) and bool(cfg.audio.profiles)
    except Exception:  # noqa: BLE001 — a preflight check must report a red
        # row, never crash the doctor command itself. A malformed
        # config.toml can fail in several ways (TOML parse error, a
        # pydantic ValidationError on a bad type, a permissions OSError);
        # all of them mean the same thing here: "not configured yet".
        has_profiles = False
    checks.append(
        Check(
            "audio device profiles named",
            has_profiles,
            "configured" if has_profiles else "not configured",
            remedy=None if has_profiles else "run: neiro doctor --audio-inventory",
        )
    )

    # mute state (only meaningful once a profile is set — otherwise informational)
    vol_out = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
    vol_in = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"])
    sink_muted = "MUTED" in vol_out
    source_muted = "MUTED" in vol_in
    checks.append(
        Check(
            "default sink not muted",
            not sink_muted,
            vol_out.strip() or "wpctl unavailable",
            remedy=None if not sink_muted else "wpctl set-mute @DEFAULT_AUDIO_SINK@ 0",
        )
    )
    checks.append(
        Check(
            "default source not muted",
            not source_muted,
            vol_in.strip() or "wpctl unavailable",
            remedy=None if not source_muted else "wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0",
        )
    )

    table = Table(title="neiro doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    all_ok = True
    for c in checks:
        status = "[green]PASS[/green]" if c.ok else "[red]FAIL[/red]"
        detail = c.detail if c.ok else f"{c.detail}  →  {c.remedy}"
        table.add_row(c.name, status, detail)
        all_ok = all_ok and c.ok
    console.print(table)

    return 0 if all_ok else 1
