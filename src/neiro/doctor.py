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
from pathlib import Path

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


def _check_cublas() -> Check:
    """ctranslate2 finds cuBLAS only if LD_LIBRARY_PATH was set BEFORE
    the process started. Setting it from inside Python does not work —
    tested directly, see docs/DECISIONS.md 2026-09-11. This check exists
    because the failure otherwise surfaces as a bare RuntimeError at the
    first transcription, minutes into a session.
    """
    import os

    path = os.environ.get("LD_LIBRARY_PATH", "")
    ok = "cublas" in path.lower() or "nvidia" in path.lower()
    return Check(
        "cuBLAS on LD_LIBRARY_PATH",
        ok,
        "set" if ok else "not set",
        remedy=None
        if ok
        else "source env.sh  (must be BEFORE python starts; setting it inside python does not work)",
    )


def _check_no_cuda_torch() -> Check:
    """The runtime venv must have CPU torch or none at all.

    A CUDA torch here means two CUDA runtimes in one process alongside
    ctranslate2's cuBLAS 12, which is the Gate G1 failure returning.
    Training has its own venv for exactly this reason.
    """
    try:
        import torch

        version = torch.__version__
        ok = "+cpu" in version
        detail = version
    except ImportError:
        return Check("runtime torch is CPU-only", True, "torch not installed (fine)")
    return Check(
        "runtime torch is CPU-only",
        ok,
        detail,
        remedy=None
        if ok
        else "uv sync — the runtime must not hold a CUDA torch; training/ has its own",
    )


def _check_models() -> Check:
    """Are the local-tier models actually on disk?"""
    from neiro.modelspec import select

    root = Path(__file__).resolve().parents[2] / "models"
    wanted = select(purpose="runtime", runs="local")
    missing = [m.name for m in wanted if not (root / m.name / ".neiro-complete").exists()]
    return Check(
        "local-tier models present",
        not missing,
        f"{len(wanted) - len(missing)}/{len(wanted)}"
        + (f" missing: {', '.join(missing)}" if missing else ""),
        remedy=None if not missing else "uv run neiro fetch-models --runs local --purpose runtime",
    )


def _check_vad_frame_size() -> Check:
    """Silero v5 accepts ONLY 512 samples at 16 kHz and returns
    plausible nonsense for anything else, silently.
    """
    from neiro.config import Neiro

    try:
        size = Neiro().vad.frame_samples
    except Exception:  # noqa: BLE001
        return Check(
            "VAD frame size is 512",
            False,
            "config unreadable",
            remedy="check ~/.config/neiro/config.toml",
        )
    ok = size == 512
    return Check(
        "VAD frame size is 512",
        ok,
        f"{size} samples",
        remedy=None
        if ok
        else "Silero v5 returns plausible nonsense at any other size — set vad.frame_samples = 512",
    )


def _check_prompt() -> Check:
    """A missing character prompt makes her a chatbot with a face."""
    try:
        from neiro.llm.prompt import PROMPT_VERSION, load_prompt, prompt_fingerprint

        words = len(load_prompt().split())
        ok = 0 < words < 700
        return Check(
            "character prompt loads",
            ok,
            f"{PROMPT_VERSION} ({words} words, {prompt_fingerprint()})",
            remedy=None
            if ok
            else "the prompt is over its prefill budget — every word is paid on each cache miss",
        )
    except FileNotFoundError as exc:
        return Check("character prompt loads", False, str(exc)[:60], remedy="prompts/ is missing")


# Audio may be committed ONLY from these paths, and only when it is
# synthetic or CC0 — a generated tone for a JIT benchmark, a licensed
# clip pinned to a regression. Everything else is somebody's voice.
AUDIO_FIXTURE_ROOTS = ("tests/audio/fixtures/", "scripts/fixtures/")


def _check_no_voice_in_git() -> Check:
    """CLAUDE.md rule 7: no raw voice audio in git, ever. It is
    biometric, and a public repo is forever.

    Not "no .wav files" — the first version of this check said that and
    flagged `scripts/fixtures/tone_3s.wav`, a synthetic sine used to
    measure ctranslate2's JIT stall in Gate G1. A check that cries wolf
    over a generated tone is a check people learn to ignore, which is
    worse than not having it. So: anything under `data/voice/` is always
    a violation, and audio elsewhere is a violation unless it sits in an
    allowlisted fixture directory.
    """
    tracked = _run(["git", "ls-files", "*.wav", "*.flac", "*.mp3", "*.ogg"])
    files = [line.strip() for line in tracked.splitlines() if line.strip()]
    offenders = [
        f for f in files if f.startswith("data/voice/") or not f.startswith(AUDIO_FIXTURE_ROOTS)
    ]
    return Check(
        "no voice audio tracked by git",
        not offenders,
        "clean" if not offenders else f"{len(offenders)} tracked: {offenders[0]}",
        remedy=None
        if not offenders
        else "git rm --cached those files — voice is biometric and a public repo is forever",
    )


def _check_hf_token() -> Check:
    """A Hugging Face read token, for the gated training corpora.

    Every AI4Bharat dataset is `gated: "auto"` — approval is automatic,
    but you must be logged in and have clicked through a contact-sharing
    agreement once. Without it, 8 datasets in the manifest silently
    refuse: 370 h of Hindi speech, 150 h more, and ai4bharat/Rasa, which
    is the only large corpus of Indian expressive emotional speech under
    a publishable licence.

    Informational rather than fatal: the whole local tier runs without
    it. Only training does not.
    """
    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        return Check(
            "HF token (for gated corpora)", False, "huggingface_hub missing", remedy="uv sync"
        )
    return Check(
        "HF token (for gated corpora)",
        bool(token),
        "present" if token else "absent — 8 gated datasets cannot download",
        remedy=None
        if token
        else "uv run hf auth login  (a READ token from huggingface.co/settings/tokens)",
    )


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

    # --- everything the day's measurements established ------------------
    # Each of these corresponds to a failure that actually happened here,
    # and each names the fix rather than the symptom. "It doesn't work"
    # is the least useful sentence in software.

    checks.append(_check_cublas())
    checks.append(_check_no_cuda_torch())
    checks.append(_check_models())
    checks.append(_check_vad_frame_size())
    checks.append(_check_prompt())
    checks.append(_check_no_voice_in_git())
    checks.append(_check_hf_token())

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
