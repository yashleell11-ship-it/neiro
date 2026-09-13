"""The preflight. Every assumption the rest of the project makes gets
checked here, with a readable remedy on failure — never a stack trace.

Stage 0 Task 0 lives here: `audio_inventory()` refuses to let the project
proceed until the two device profiles (earbuds / speakers) are named
explicitly in config.toml, because this laptop's PipeWire 'default'
sink/source silently changes and a voice assistant that follows it will
one day play into a muted device and look broken.

Stage 0 Task 18 made this the one consolidated preflight. Every row is a
small pure function over facts that were already observed —
`check_gpu(gpu, need_mib)`, not "read nvidia-smi and decide" — and the
`_observe_*` functions are the only code that touches the machine. The
split is what makes each row provable: a test hands a row function a
fake that should go red and watches it go red. A row that cannot be made
to fail in a test is a row nobody can trust to fail for real, and a
preflight that cannot fail is a decoration.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

# The runtime's own definition of "the model is thinking", so the doctor
# and the per-turn guard can never disagree about what a reasoning reply
# looks like — the field names were measured, not assumed (see
# llm/openai_compat.py's module docstring).
from neiro.llm.ollama_native import THINKING_FIELD
from neiro.llm.openai_compat import REASONING_FIELDS, THINK_SNIFF_CHARS

console = Console()

# The marker fetch_models.py writes when a snapshot finished. Spelled
# here as well because the doctor must not import a script.
MODEL_COMPLETE_MARKER = ".neiro-complete"

# CTranslate2's own support table: the float16 and int8_float16 kernels
# need Compute Capability >= 7.0. Below that the wanted compute type is
# silently downgraded — a fact about the library, not a tunable.
MIN_COMPUTE_CAPABILITY = 7.0

# Measured 2026-09-13 against the CPU-only ollama on this laptop: a cold
# load of the 4B took over 30 s and a warm one-token reply 8 s, nearly
# all of it prompt eval. The read timeout survives a cold load; a server
# that is simply absent still fails at connect, which is separate.
LLM_PROBE_TIMEOUT_S = 60.0
LLM_PROBE_CONNECT_S = 5.0

# The one-token probe's prompt. Short keeps prompt eval cheap on CPU.
LLM_PROBE_PROMPT = "hi"

# The tools the Justfile, the tests and web/ build with. ruff and pytest
# are installed by uv into the venv (not on PATH), bun and just come from
# pacman — both are in [extra], checked with `pacman -Si` (CLAUDE.md 3).
DEV_TOOLS = ("ruff", "pytest", "bun", "just")
VENV_TOOLS = ("ruff", "pytest")

# The components whose weights sit in VRAM on the local tier: the LLM
# under llama-server / ollama and STT under ctranslate2. Kokoro runs on
# CPU torch; VAD and the endpointer are onnx on CPU.
VRAM_COMPONENTS = ("llm", "stt")

# Hashing multi-GB weights: 1 MiB reads keep it disk-bound, not Python-bound.
HASH_CHUNK_BYTES = 1 << 20


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str | None = None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
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


# --- the rows ---------------------------------------------------------
# Each takes facts and returns a verdict. Nothing below this line until
# the observers reads the machine, so every row can be driven to red.


def check_python(version: tuple[int, int]) -> Check:
    """pyproject pins `>=3.12,<3.13`; the CPU torch wheel and ctranslate2
    are resolved for exactly that, and a 3.13 venv resolves to nothing.
    """
    ok = version == (3, 12)
    return Check(
        "Python 3.12",
        ok,
        f"{version[0]}.{version[1]}",
        remedy=None if ok else "uv python pin 3.12 && uv sync",
    )


def check_toolchain(found: Mapping[str, str | None]) -> Check:
    """`just`, `bun`, `ruff`, `pytest` — the four things a task is run
    and verified with. `just` is the one people hit: the justfile's own
    header says it is not installed yet (it needs sudo), so this row is
    what turns "just doctor: command not found" into the pacman line.
    """
    missing = [tool for tool in DEV_TOOLS if not found.get(tool)]
    remedies = []
    if any(tool in VENV_TOOLS for tool in missing):
        remedies.append("uv sync  (ruff and pytest come from the dev group)")
    system = [tool for tool in missing if tool not in VENV_TOOLS]
    if system:
        remedies.append(f"sudo pacman -S {' '.join(system)}  (in [extra], checked with pacman -Si)")
    return Check(
        "dev toolchain (ruff, pytest, bun, just)",
        not missing,
        "all present" if not missing else f"missing: {', '.join(missing)}",
        remedy="; ".join(remedies) or None,
    )


def check_ct2_compute_type(supported: Iterable[str] | None, wanted: str, error: str = "") -> Check:
    """Gate G1 proved `int8_float16` on this GPU (docs/DECISIONS.md
    2026-09-11). ctranslate2 does not fail when asked for a compute type
    the device lacks — it downgrades silently, and the STT gets slower
    with no message anywhere. So the doctor asks the library what the
    device offers and holds it to config's `stt.compute_type`.
    """
    name = f"ctranslate2 offers {wanted} on cuda"
    if supported is None:
        return Check(
            name,
            False,
            error[:80] or "ctranslate2 unavailable",
            remedy="uv sync — ctranslate2>=4.7.0 built for CUDA 12 — and nvidia-smi must see the GPU",
        )
    offered = sorted(supported)
    ok = wanted in offered
    return Check(
        name,
        ok,
        f"offered: {', '.join(offered)}" if ok else f"not offered; cuda has {', '.join(offered)}",
        remedy=None
        if ok
        else "set stt.compute_type in ~/.config/neiro/config.toml to one that is offered, "
        "or fix the driver — int8 on cuda needs ctranslate2>=4.7.0 (docs/CORRECTIONS.md)",
    )


@dataclass(frozen=True)
class Gpu:
    free_mib: int
    total_mib: int
    compute_cap: float


def check_gpu(gpu: Gpu | None, need_mib: int) -> Check:
    """Free VRAM against what the tier's weights occupy, and the compute
    capability the STT compute type needs.

    `need_mib` is derived from modelspec — the llm + stt weights of the
    runtime tier — not chosen. It is a floor: KV cache and activations
    come on top. Free VRAM is the number that moves: a training run in
    the other venv holds 2-4 GB and the doctor is the first thing to
    say so, rather than the STT failing to allocate mid-sentence.
    """
    name = "GPU: free VRAM and compute capability"
    if gpu is None:
        return Check(
            name,
            False,
            "nvidia-smi gave nothing",
            remedy="the NVIDIA driver must expose the GPU — `nvidia-smi` should list it",
        )
    detail = f"{gpu.free_mib}/{gpu.total_mib} MiB free, cc {gpu.compute_cap}"
    if gpu.compute_cap < MIN_COMPUTE_CAPABILITY:
        return Check(
            name,
            False,
            f"{detail} — below {MIN_COMPUTE_CAPABILITY}",
            remedy="this GPU has no fp16 tensor cores; int8_float16 cannot run here — "
            "set stt.compute_type = 'int8' or use the box tier",
        )
    if gpu.free_mib < need_mib:
        return Check(
            name,
            False,
            f"{detail}; the tier's llm+stt weights need {need_mib} MiB",
            remedy="free VRAM: `nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv` names who holds it (a training run holds 2-4 GB)",
        )
    return Check(name, True, f"{detail}; tier needs {need_mib} MiB")


@dataclass(frozen=True)
class LlmProbe:
    """What one-token request to the LLM server came back with."""

    server: str | None  # "ollama" | "llama-server" | "unknown" | None = nothing answered
    model: str
    content: str = ""
    reasoning: str = ""  # anything in a reasoning field — must be empty
    error: str = ""


_THINKING_REMEDY = (
    "thinking must be OFF and asserted (CLAUDE.md 4): ollama needs `think: false` on "
    "/api/chat — it is ignored on /v1; llama-server needs "
    "`--reasoning-budget 0 --reasoning-format none`. Verify with curl, never assume"
)


def check_llm(probe: LlmProbe, base_url: str) -> Check:
    """The server is up AND one token came back as words.

    Not just "the port answers": the failure measured on 2026-09-13 was a
    server that answered every request perfectly and spent the whole
    budget on reasoning while `content` stayed empty. So the probe asks
    for one token with thinking off and the row fails on the same three
    signatures the per-turn guard fails on — a `<think` opener, a
    populated reasoning field, and a reply that carried no content at all.
    """
    name = "LLM server up, 1 token back, thinking off"
    if probe.server is None:
        return Check(
            name,
            False,
            f"nothing answered at {base_url}",
            remedy="start it: `ollama serve` (native /api) or `llama-server --reasoning-budget 0 "
            "--reasoning-format none`, or point llm.base_url in ~/.config/neiro/config.toml at it",
        )
    if probe.error:
        model_missing = "not found" in probe.error.lower()
        return Check(
            name,
            False,
            f"{probe.server}: {probe.error[:80]}",
            remedy=f"`ollama list` does not have {probe.model!r} — pull or create it, or set "
            "llm.model in ~/.config/neiro/config.toml to a name it shows"
            if model_missing
            else "the server answered but refused the request — read the message",
        )
    if probe.reasoning or "<think" in probe.content[:THINK_SNIFF_CHARS].lower():
        return Check(
            name,
            False,
            f"{probe.server} is reasoning: content={probe.content[:20]!r} "
            f"reasoning={probe.reasoning[:20]!r}",
            remedy=_THINKING_REMEDY,
        )
    if not probe.content:
        return Check(
            name,
            False,
            f"{probe.server} returned no content for 1 token — reasoning may be hidden under "
            "a field this check does not know",
            remedy=_THINKING_REMEDY,
        )
    return Check(name, True, f"{probe.server} @ {base_url}, {probe.model}: {probe.content!r}")


@dataclass(frozen=True)
class NodeState:
    """One PipeWire node as `pactl list` reports it. The id is the
    PipeWire object id — the same number `wpctl` takes, so the remedy
    can be pasted."""

    id: str
    name: str
    kind: str  # "sink" | "source"
    muted: bool


def parse_pactl_list(kind: str, text: str) -> list[NodeState]:
    """`pactl list sinks` / `pactl list sources` → nodes with mute state.

    The short form has no mute column; the long form is blocks headed
    `Sink #60` with tab-indented `Name:` and `Mute:` lines.
    """
    header = f"{kind.capitalize()} #"
    nodes: list[NodeState] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if current.get("id") and current.get("name"):
            nodes.append(
                NodeState(
                    id=current["id"],
                    name=current["name"],
                    kind=kind,
                    muted=current.get("mute", "no").strip().lower() == "yes",
                )
            )

    for line in text.splitlines():
        if line.startswith(header):
            flush()
            current = {"id": line[len(header) :].strip()}
        elif line.strip().startswith("Name:"):
            current["name"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("Mute:"):
            current["mute"] = line.split(":", 1)[1]
    flush()
    return nodes


# Which PipeWire object class a profile's device names.
_PROFILE_KIND = {"input": "source", "output": "sink"}


def check_profile_node(
    profile: str, kind: str, wanted: str | None, nodes: Iterable[NodeState]
) -> Check:
    """The active profile's named node is live and not muted.

    Task 0's whole point was to name devices rather than follow
    'default'; this is the row that catches the other half — the named
    device being absent (earbuds not connected) or muted (the built-in
    card, most days). The remedy is the exact `wpctl` line, by object
    id, because "unmute it" sends people to a mixer that shows
    descriptions, not node names.
    """
    name = f"profile {kind} node live & unmuted"
    if not profile:
        return Check(name, False, "no active profile", remedy="run: neiro doctor --audio-inventory")
    if not wanted:
        return Check(
            name,
            False,
            f"[audio.profiles.{profile}] has no {kind}_device",
            remedy="set it to a node name from: neiro doctor --audio-inventory",
        )
    pw_kind = _PROFILE_KIND[kind]
    node = next((n for n in nodes if n.kind == pw_kind and n.name == wanted), None)
    if node is None:
        return Check(
            name,
            False,
            f"{profile}: {wanted!r} is not a live {pw_kind}",
            remedy="connect the device, or set [audio] active_profile to a profile whose "
            "nodes are live — neiro doctor --audio-inventory shows them",
        )
    if node.muted:
        return Check(
            name,
            False,
            f"{profile}: #{node.id} {wanted} is MUTED",
            remedy=f"wpctl set-mute {node.id} 0",
        )
    return Check(name, True, f"{profile}: #{node.id} {wanted}")


def check_samplerates(
    input_rate: int, output_rate: int, input_error: str | None, output_error: str | None, via: str
) -> Check:
    """The configured rates open on the configured device.

    16 kHz in is what Silero and Whisper consume; 24 kHz out is what
    Kokoro produces and the browser's AudioContext is told to expect.
    PipeWire resamples anything, so a refusal here means the stream is
    not reaching PipeWire at all — which is what the `mic-test`
    chipmunk playback was built to make audible, and what this row
    makes visible before any audio is opened.
    """
    name = f"{input_rate} Hz in / {output_rate} Hz out accepted"
    problems = [
        f"{label} {rate} Hz refused: {error}"
        for label, rate, error in (
            ("input", input_rate, input_error),
            ("output", output_rate, output_error),
        )
        if error
    ]
    if not problems:
        return Check(name, True, f"via {via}")
    no_pulse = any(
        "no input device matching" in p.lower() or "no output device matching" in p.lower()
        for p in problems
    )
    return Check(
        name,
        False,
        "; ".join(problems)[:120],
        remedy="PipeWire's pulse server is not running: systemctl --user start pipewire-pulse"
        if no_pulse
        else "PipeWire resamples anything, so the stream is not reaching it — check "
        "[audio.profiles] node names and docs/ARCHITECTURE.md's device routing",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(etag: str) -> bool:
    return len(etag) == 64 and all(c in "0123456789abcdef" for c in etag)


def expected_hashes(model_dir: Path) -> dict[str, str]:
    """relative file → etag, from the metadata huggingface_hub writes
    beside every file it downloads into a `local_dir`.

    modelspec carries no hashes of its own — the Hub verifies its own
    files (training/manifest.py says the same) — but it leaves the
    receipt: for an LFS file the etag IS the sha256 of the content; for
    a small non-LFS file it is a git blob id, which is presence-only.
    """
    cache = model_dir / ".cache" / "huggingface" / "download"
    if not cache.is_dir():
        return {}
    hashes: dict[str, str] = {}
    for meta in sorted(cache.rglob("*.metadata")):
        lines = meta.read_text(encoding="utf-8", errors="replace").splitlines()
        etag = lines[1].strip() if len(lines) > 1 else ""
        hashes[str(meta.relative_to(cache).with_suffix(""))] = etag
    return hashes


def verify_model_dir(model_dir: Path, digest: Callable[[Path], str] = _sha256_file) -> list[str]:
    """Everything wrong with one downloaded model, in words; empty means
    every file is present and every LFS file hashes to what the Hub
    said it should."""
    if not (model_dir / MODEL_COMPLETE_MARKER).exists():
        return ["not downloaded"]
    expected = expected_hashes(model_dir)
    if not expected:
        return ["no download metadata to verify against"]
    problems = []
    for relative, etag in expected.items():
        target = model_dir / relative
        if not target.is_file():
            problems.append(f"{relative} missing")
        elif _is_sha256(etag) and digest(target) != etag:
            problems.append(f"{relative} sha256 mismatch")
    return problems


def check_model_files(
    wanted: Iterable[str], root: Path, digest: Callable[[Path], str] = _sha256_file
) -> Check:
    """Every model the tier needs is on disk with matching hashes.

    Presence was the old row; a marker file is written after the
    snapshot returns, and a snapshot that resumed a partial file three
    times can return a file that is the right length and the wrong
    bytes. Hashing 4.6 GB costs ~10 s cold on this laptop — printed, so
    the cost is never a surprise. Nothing is downloaded here; the
    remedy names the command that does.
    """
    started = time.perf_counter()
    names = list(wanted)
    problems = {name: verify_model_dir(root / name, digest) for name in names}
    bad = {name: found for name, found in problems.items() if found}
    elapsed = time.perf_counter() - started
    verified = len(names) - len(bad)
    if not bad:
        return Check(
            "local-tier models present & sha256-verified",
            True,
            f"{verified}/{len(names)} verified in {elapsed:.1f} s",
        )
    first_name, first_problems = next(iter(bad.items()))
    more = len(bad) - 1
    detail = f"{verified}/{len(names)}; {first_name}: {first_problems[0]}" + (
        f" (+{more} more)" if more else ""
    )
    undownloaded = [name for name, found in bad.items() if found == ["not downloaded"]]
    if undownloaded:
        remedy = "uv run neiro fetch-models --runs local --purpose runtime"
    else:
        remedy = (
            f"rm the named file under models/{first_name}/, then: "
            f"uv run neiro fetch-models --only {first_name} --force  (HF re-pulls only what is missing)"
        )
    return Check("local-tier models present & sha256-verified", False, detail, remedy=remedy)


# Audio may be committed ONLY from these paths, and only when it is
# synthetic or CC0 — a generated tone for a JIT benchmark, a licensed
# clip pinned to a regression. Everything else is somebody's voice.
AUDIO_FIXTURE_ROOTS = ("tests/audio/fixtures/", "scripts/fixtures/")

# What `git ls-files` is asked for. Not only .wav: a voice is a voice in
# any container.
AUDIO_GLOBS = ("*.wav", "*.flac", "*.mp3", "*.ogg")


def check_no_voice_in_git(tracked: Iterable[str]) -> Check:
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
    files = [line.strip() for line in tracked if line.strip()]
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


@dataclass(frozen=True)
class WsOutcome:
    """What happened when a peer without a token, then one with it,
    knocked on the WebSocket."""

    unauthenticated_accepted: bool
    close_code: int | None
    authenticated_hello: bool
    error: str = ""


def check_ws_auth(outcome: WsOutcome) -> Check:
    """A localhost WebSocket is not private (server.py's docstring), so
    the wall is checked from outside every time, not trusted.

    Both directions, because a server that refuses everyone passes
    "refuses the unauthenticated" and the face could never connect. The
    PEP 563 regression looked exactly like that: every handshake 403,
    valid token or not, and no error anywhere.
    """
    name = "WebSocket refuses unauthenticated peers"
    if outcome.error:
        return Check(
            name,
            False,
            outcome.error[:80],
            remedy="the endpoint could not be driven at all — see tests/test_ws_auth.py",
        )
    if outcome.unauthenticated_accepted:
        return Check(
            name,
            False,
            "a peer with NO token got a socket",
            remedy="server.py: Session.check must run before accept() — see tests/test_ws_auth.py",
        )
    if not outcome.authenticated_hello:
        return Check(
            name,
            False,
            f"refused without a token (close {outcome.close_code}) but the right token got no hello",
            remedy="the face could never connect — is the handler's `WebSocket` annotation a real "
            "type (no PEP 563 in server.py)? see tests/test_ws_auth.py",
        )
    return Check(name, True, f"closed {outcome.close_code} without a token; hello with it")


# --- rows kept from the day's measurements -------------------------------
# Each of these corresponds to a failure that actually happened here,
# and each names the fix rather than the symptom. "It doesn't work"
# is the least useful sentence in software.


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


def _check_vad_frame_size(cfg) -> Check:
    """Silero v5 accepts ONLY 512 samples at 16 kHz and returns
    plausible nonsense for anything else, silently.
    """
    if cfg is None:
        return Check(
            "VAD frame size is 512",
            False,
            "config unreadable",
            remedy="check ~/.config/neiro/config.toml",
        )
    size = cfg.vad.frame_samples
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


def _check_profiles_named(cfg) -> Check:
    has_profiles = cfg is not None and bool(cfg.audio.active_profile) and bool(cfg.audio.profiles)
    return Check(
        "audio device profiles named",
        has_profiles,
        "configured" if has_profiles else "not configured",
        remedy=None if has_profiles else "run: neiro doctor --audio-inventory",
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


# --- observers: the only code that reads the machine ---------------------


def _load_config():
    """`Neiro()` or None. A malformed config.toml can fail in several
    ways (TOML parse error, a pydantic ValidationError on a bad type, a
    permissions OSError); all of them mean the same thing to every row
    that needs it: "not configured yet" — a red row, never a crash of
    the doctor itself.
    """
    from neiro.config import Neiro

    try:
        return Neiro()
    except Exception:  # noqa: BLE001 — see above
        return None


def _find_tools() -> dict[str, str | None]:
    # sys.prefix is the venv; sys.executable resolved would be uv's
    # python, whose bin/ holds no ruff.
    venv_bin = Path(sys.prefix) / "bin"
    found: dict[str, str | None] = {}
    for tool in DEV_TOOLS:
        in_venv = venv_bin / tool
        found[tool] = str(in_venv) if in_venv.exists() else shutil.which(tool)
    return found


def _observe_ct2() -> tuple[list[str] | None, str]:
    try:
        import ctranslate2

        return list(ctranslate2.get_supported_compute_types("cuda")), ""
    except Exception as exc:  # noqa: BLE001 — the message is the row
        return None, f"{type(exc).__name__}: {exc}"


def _observe_gpu() -> Gpu | None:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    first = out.strip().splitlines()[0] if out.strip() else ""
    try:
        free, total, cap = (part.strip() for part in first.split(","))
        return Gpu(free_mib=int(free), total_mib=int(total), compute_cap=float(cap))
    except ValueError:
        return None


def _vram_need_mib() -> int:
    from neiro.modelspec import select

    weights_gb = sum(
        m.size_gb for m in select(purpose="runtime", runs="local") if m.component in VRAM_COMPONENTS
    )
    return int(weights_gb * 1e9 / (1 << 20))


def _detect_llm_server(client, base_url: str) -> str | None:
    import httpx

    try:
        if client.get(f"{base_url}/api/version").status_code == 200:
            return "ollama"
        if client.get(f"{base_url}/health").status_code == 200:
            return "llama-server"
        return "unknown"
    except httpx.HTTPError:
        return None


def _probe_llm(base_url: str, model: str) -> LlmProbe:
    """One token, thinking off, from whichever server is listening.

    ollama is asked on its native /api/chat because its /v1 endpoint
    ignores every thinking switch (llm/ollama_native.py); llama-server
    is asked on /v1 with the `chat_template_kwargs` the plan specifies.
    """
    import httpx

    messages = [{"role": "user", "content": LLM_PROBE_PROMPT}]
    timeout = httpx.Timeout(LLM_PROBE_TIMEOUT_S, connect=LLM_PROBE_CONNECT_S)
    with httpx.Client(timeout=timeout) as client:
        server = _detect_llm_server(client, base_url)
        if server is None:
            return LlmProbe(None, model)
        if server == "unknown":
            return LlmProbe(
                server,
                model,
                error="answered, but is neither ollama (/api/version) nor llama-server (/health)",
            )
        try:
            if server == "ollama":
                response = client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {"num_predict": 1},
                    },
                )
                data = response.json()
                if response.status_code != 200 or data.get("error"):
                    return LlmProbe(
                        server,
                        model,
                        error=str(data.get("error") or f"HTTP {response.status_code}"),
                    )
                message = data.get("message") or {}
                return LlmProbe(
                    server,
                    model,
                    content=message.get("content") or "",
                    reasoning=message.get(THINKING_FIELD) or "",
                )
            response = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            data = response.json()
            if response.status_code != 200:
                return LlmProbe(
                    server, model, error=str(data.get("error") or f"HTTP {response.status_code}")
                )
            message = (data.get("choices") or [{}])[0].get("message") or {}
            return LlmProbe(
                server,
                model,
                content=message.get("content") or "",
                reasoning="".join(str(message.get(key) or "") for key in REASONING_FIELDS),
            )
        except (httpx.HTTPError, ValueError) as exc:
            return LlmProbe(server, model, error=f"{type(exc).__name__}: {exc}")


def _pipewire_nodes() -> list[NodeState]:
    return parse_pactl_list("sink", _run(["pactl", "list", "sinks"])) + parse_pactl_list(
        "source", _run(["pactl", "list", "sources"])
    )


def _observe_samplerates(cfg) -> tuple[str | None, str | None, str]:
    """Ask PortAudio whether the configured rates open on 'pulse' aimed
    at the active profile — `check_*_settings` asks without opening a
    stream, so nothing plays and nothing records."""
    from neiro.audio.devices import (
        NoActiveProfileError,
        apply_to_environment,
        resolve_active_profile,
        suppress_alsa_errors,
    )

    try:
        device = resolve_active_profile(cfg)
        apply_to_environment(device)
        via = f"{device.portaudio_device} → {device.profile_name}"
    except NoActiveProfileError:
        via = "pulse (PipeWire's default — no active profile)"

    suppress_alsa_errors()
    import sounddevice as sd

    def attempt(check: Callable[..., None], rate: int) -> str | None:
        try:
            check(device="pulse", samplerate=rate, channels=1, dtype="float32")
            return None
        except Exception as exc:  # noqa: BLE001 — PortAudioError or ValueError, both are the row
            return str(exc)[:60]

    return (
        attempt(sd.check_input_settings, cfg.audio.input_samplerate),
        attempt(sd.check_output_settings, cfg.audio.output_samplerate),
        via,
    )


def _tracked_audio() -> list[str]:
    return _run(["git", "ls-files", *AUDIO_GLOBS]).splitlines()


def _observe_ws() -> WsOutcome:
    """Drive the real app through Starlette's in-process client: no
    port, no uvicorn, no server left behind if the doctor is killed."""
    import warnings

    with warnings.catch_warnings():
        # starlette 1.x deprecates driving its test client over httpx;
        # a preflight's output is the table, not a library's roadmap.
        warnings.simplefilter("ignore")
        from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from neiro.server import TOKEN_SUBPROTOCOL_PREFIX, Session, build_app

    session = Session()
    client = TestClient(build_app(session))
    origin = {"Origin": "http://127.0.0.1:8760"}

    accepted = False
    code: int | None = None
    try:
        with client.websocket_connect("/neiro", headers=origin):
            accepted = True
    except WebSocketDisconnect as exc:
        code = exc.code

    hello = False
    try:
        with client.websocket_connect(
            "/neiro", subprotocols=[TOKEN_SUBPROTOCOL_PREFIX + session.token], headers=origin
        ) as ws:
            hello = json.loads(ws.receive_text()).get("t") == "hello"
    except Exception as exc:  # noqa: BLE001 — a refusal here is the finding, not a crash
        return WsOutcome(accepted, code, False, error=f"with the right token: {exc}")
    return WsOutcome(accepted, code, hello)


def _row(name: str, make: Callable[[], Check]) -> Check:
    """A check that throws becomes a red row with the exception in it.
    A traceback from one row would hide every other row's verdict."""
    try:
        return make()
    except Exception as exc:  # noqa: BLE001 — see above
        return Check(
            name,
            False,
            f"{type(exc).__name__}: {str(exc)[:70]}",
            remedy="this check itself failed — the message above is the lead",
        )


def gather_checks() -> list[Check]:
    """Every row, in the order the fixes should be applied: the
    interpreter and tools first, then the GPU stack, then the models,
    then what talks to the outside — a red at the top usually explains
    the reds below it."""
    from neiro.config import LlmConfig, SttConfig
    from neiro.modelspec import select

    cfg = _load_config()
    stt = cfg.stt if cfg is not None else SttConfig()
    llm = cfg.llm if cfg is not None else LlmConfig()
    models_root = Path(__file__).resolve().parents[2] / "models"
    active = cfg.audio.active_profile if cfg is not None else ""
    profile = cfg.audio.profiles.get(active) if cfg is not None else None

    def node_row(kind: str) -> Callable[[], Check]:
        wanted = getattr(profile, f"{kind}_device", None) if profile else None
        return lambda: check_profile_node(active, kind, wanted, _pipewire_nodes())

    def ct2_row() -> Check:
        supported, error = _observe_ct2()
        return check_ct2_compute_type(supported, stt.compute_type, error)

    def samplerate_row() -> Check:
        if cfg is None:
            return Check(
                "audio rates accepted",
                False,
                "config unreadable",
                remedy="check ~/.config/neiro/config.toml",
            )
        return check_samplerates(
            cfg.audio.input_samplerate, cfg.audio.output_samplerate, *_observe_samplerates(cfg)
        )

    return [
        check_python(sys.version_info[:2]),
        _row("dev toolchain", lambda: check_toolchain(_find_tools())),
        _check_cublas(),
        _row("ctranslate2 compute type", ct2_row),
        _row("GPU", lambda: check_gpu(_observe_gpu(), _vram_need_mib())),
        _check_no_cuda_torch(),
        _row(
            "local-tier models",
            lambda: check_model_files(
                [m.name for m in select(purpose="runtime", runs="local")], models_root
            ),
        ),
        _check_vad_frame_size(cfg),
        _check_prompt(),
        _row(
            "LLM server",
            lambda: check_llm(_probe_llm(llm.base_url.rstrip("/"), llm.model), llm.base_url),
        ),
        _check_profiles_named(cfg),
        _row("profile input node", node_row("input")),
        _row("profile output node", node_row("output")),
        _row("audio rates accepted", samplerate_row),
        _row("WebSocket auth", lambda: check_ws_auth(_observe_ws())),
        _row("no voice audio in git", lambda: check_no_voice_in_git(_tracked_audio())),
        _check_hf_token(),
    ]


def render(checks: list[Check]) -> int:
    """One line per row, red rows carry their fix, and the summary is
    red if any row is — the exit code says the same thing to a script."""
    table = Table(title="neiro doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for c in checks:
        status = "[green]PASS[/green]" if c.ok else "[red]FAIL[/red]"
        detail = c.detail if c.ok else f"{c.detail}  →  {c.remedy}"
        # Escaped, because a fix is pasted, not styled: rich read
        # `[audio]` and `[extra]` as markup and printed neither, and the
        # first real run told people to set " active_profile".
        table.add_row(escape(c.name), status, escape(detail))
    console.print(table)

    failed = [c for c in checks if not c.ok]
    if failed:
        console.print(
            f"[red]FAIL[/red] {len(failed)} of {len(checks)} checks — each red row names its fix"
        )
        return 1
    console.print(f"[green]PASS[/green] all {len(checks)} checks")
    return 0


def run_doctor(
    audio_inventory_only: bool = False, *, gather: Callable[[], list[Check]] | None = None
) -> int:
    """What `neiro doctor` calls. `gather` is the seam a test uses to
    hand it rows without touching the machine."""
    if audio_inventory_only:
        return audio_inventory()
    return render((gather or gather_checks)())
