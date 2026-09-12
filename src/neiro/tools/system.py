"""GREEN-tier system readouts, and the YELLOW-tier knobs next to them.

Deliberately no psutil: everything here comes from stdlib plus CLI tools
already installed on this machine. One less dependency, and reading
/proc directly is both faster and more obvious about where a number came
from.

Note on brightness (a real finding from the planning research): the only
backlight device on this laptop is `nvidia_wmi_ec_backlight`, and
because the GPU MUX is in discrete mode, writing to it *succeeds and
does nothing* — the panel is driven by the dGPU. Yash already worked
around this with ~/.local/bin/brightness-smart.sh, which dims via
hyprsunset gamma instead. So `adjust_brightness` routes through that
script rather than sysfs. A tool that returns 0 and changes nothing is
exactly the silent failure this project is built to avoid.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

BRIGHTNESS_SCRIPT = Path.home() / ".local/bin/brightness-smart.sh"


def _run(cmd: list[str], timeout: float = 2.0) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


@dataclass(frozen=True)
class SystemStats:
    battery_percent: int | None
    battery_charging: bool | None
    cpu_percent: float | None
    memory_percent: float | None
    disk_free_gb: float | None

    def describe(self) -> str:
        """One spoken sentence. Short — she's talking, not printing a
        dashboard.
        """
        parts = []
        if self.battery_percent is not None:
            state = "charging" if self.battery_charging else "on battery"
            parts.append(f"battery {self.battery_percent} percent, {state}")
        if self.memory_percent is not None:
            parts.append(f"memory {self.memory_percent:.0f} percent")
        if self.disk_free_gb is not None:
            parts.append(f"{self.disk_free_gb:.0f} gigs free")
        return ", ".join(parts) if parts else "I can't read the system stats."


def _battery() -> tuple[int | None, bool | None]:
    for base in sorted(Path("/sys/class/power_supply").glob("BAT*")):
        try:
            capacity = int((base / "capacity").read_text().strip())
            status = (base / "status").read_text().strip().lower()
            return capacity, status in ("charging", "full")
        except (OSError, ValueError):
            continue
    return None, None


def _memory_percent() -> float | None:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            value = rest.strip().split()
            if value:
                info[key] = int(value[0])
        total = info.get("MemTotal")
        available = info.get("MemAvailable")
        if total and available is not None:
            return (total - available) / total * 100.0
    except (OSError, ValueError):
        pass
    return None


def _cpu_percent() -> float | None:
    """Instantaneous load as a rough percentage of all cores.

    Uses loadavg rather than sampling /proc/stat twice — a voice
    assistant answering "is my CPU busy" doesn't need a precise number,
    and sampling twice would mean sleeping, which the turn budget can't
    spare.
    """
    try:
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
        cores = len(
            [
                line
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("processor")
            ]
        )
        if cores:
            return min(100.0, load1 / cores * 100.0)
    except (OSError, ValueError, IndexError):
        pass
    return None


def get_system_stats(disk_path: str = str(Path.home())) -> SystemStats:
    """GREEN tier. No arguments the model can influence, no side effects."""
    percent, charging = _battery()
    try:
        usage = shutil.disk_usage(disk_path)
        disk_free_gb = usage.free / (1024**3)
    except OSError:
        disk_free_gb = None
    return SystemStats(
        battery_percent=percent,
        battery_charging=charging,
        cpu_percent=_cpu_percent(),
        memory_percent=_memory_percent(),
        disk_free_gb=disk_free_gb,
    )


@dataclass(frozen=True)
class AudioState:
    volume_percent: int | None
    muted: bool | None

    def describe(self) -> str:
        if self.volume_percent is None:
            return "I can't read the volume."
        if self.muted:
            return f"volume {self.volume_percent} percent, muted"
        return f"volume {self.volume_percent} percent"


_VOLUME = re.compile(r"Volume:\s*([0-9.]+)")


def get_audio_state() -> AudioState:
    """GREEN tier."""
    out = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
    if not out:
        return AudioState(volume_percent=None, muted=None)
    match = _VOLUME.search(out)
    volume = round(float(match.group(1)) * 100) if match else None
    return AudioState(volume_percent=volume, muted="MUTED" in out)


def set_volume(percent: int) -> str:
    """YELLOW tier. `percent` is a plain int from a constrained range —
    no string reaches a shell.

    The `-l 1.0` cap is a hard physical safety net: a misheard "volume
    one thousand" cannot blow the speakers or Yash's ears out, because
    wpctl refuses to exceed 100% regardless of what we ask for.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"volume must be 0-100, got {percent}")
    _run(["wpctl", "set-volume", "-l", "1.0", "@DEFAULT_AUDIO_SINK@", f"{percent}%"])
    return f"volume {percent} percent"


def set_mute(state: str) -> str:
    """YELLOW tier. `state` is a Literal enum member, never free text."""
    mapping = {"on": "1", "off": "0", "toggle": "toggle"}
    if state not in mapping:
        raise ValueError(f"mute state must be one of {list(mapping)}, got {state!r}")
    _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", mapping[state]])
    return f"mute {state}"


MAX_BRIGHTNESS_STEPS = 6  # 6 x 5% = 30%, plenty for one spoken request


def adjust_brightness(direction: str, steps: int = 1) -> str:
    """YELLOW tier. Routes through brightness-smart.sh, NOT sysfs.

    RELATIVE, not absolute — and that's dictated by the tool that
    actually works, not by preference. brightness-smart.sh takes only
    `-i` / `-d` and moves in fixed 5% steps; it has no "set to N"
    interface. (Checked the script rather than assuming: an earlier
    draft of this module called it with `set <percent>`, which would
    have exited 1 with a usage error every time.)

    It also happens to match how anyone actually asks for this out loud
    — "a bit dimmer", not "set brightness to 55 percent".

    The script's own floor (GAMMA_MIN=15) means a misheard "much dimmer"
    cannot black the panel out and leave Yash unable to see how to undo
    it.
    """
    flags = {"up": "-i", "down": "-d"}
    if direction not in flags:
        raise ValueError(f"direction must be one of {list(flags)}, got {direction!r}")
    if not 1 <= steps <= MAX_BRIGHTNESS_STEPS:
        raise ValueError(f"steps must be 1-{MAX_BRIGHTNESS_STEPS}, got {steps}")
    if not BRIGHTNESS_SCRIPT.exists():
        raise FileNotFoundError(
            f"{BRIGHTNESS_SCRIPT} not found — it's the only thing that actually "
            "moves this panel (the sysfs backlight is a no-op with the GPU MUX "
            "in discrete mode)."
        )
    for _ in range(steps):
        _run([str(BRIGHTNESS_SCRIPT), flags[direction]])
    return f"brightness {direction} {steps * 5} percent"
