"""`neiro doctor` — every row can be driven red.

Each row is a pure function over facts; these tests hand each one a
fake that should pass and a fake that should fail, and read the verdict
and the fix. Nothing here touches nvidia-smi, PipeWire, a model file
bigger than a few bytes, or an LLM server. The observers are the seam;
the one real probe that runs is the WebSocket one, because it is
in-process and the row it feeds would otherwise never be exercised
against the actual app.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from rich.console import Console

from neiro import doctor
from neiro.doctor import (
    Check,
    Gpu,
    LlmProbe,
    WsOutcome,
    check_ct2_compute_type,
    check_gpu,
    check_llm,
    check_model_files,
    check_no_voice_in_git,
    check_profile_node,
    check_python,
    check_samplerates,
    check_toolchain,
    check_ws_auth,
    expected_hashes,
    parse_pactl_list,
    render,
    run_doctor,
    verify_model_dir,
)


class TestPython:
    def test_312_passes(self) -> None:
        row = check_python((3, 12))
        assert row.ok and row.detail == "3.12"

    @pytest.mark.parametrize("version", [(3, 11), (3, 13)])
    def test_any_other_minor_fails_and_names_uv(self, version: tuple[int, int]) -> None:
        row = check_python(version)
        assert not row.ok
        assert "uv python pin 3.12" in row.remedy


ALL_TOOLS = {
    "ruff": "/v/bin/ruff",
    "pytest": "/v/bin/pytest",
    "bun": "/usr/bin/bun",
    "just": "/usr/bin/just",
}


class TestToolchain:
    def test_all_present_passes(self) -> None:
        assert check_toolchain(ALL_TOOLS).ok

    def test_a_missing_system_tool_names_pacman_not_uv(self) -> None:
        row = check_toolchain({**ALL_TOOLS, "just": None})
        assert not row.ok
        assert "just" in row.detail
        assert "pacman -S just" in row.remedy
        assert "uv sync" not in row.remedy

    def test_a_missing_venv_tool_names_uv_not_pacman(self) -> None:
        row = check_toolchain({**ALL_TOOLS, "ruff": None})
        assert not row.ok
        assert "uv sync" in row.remedy
        assert "pacman" not in row.remedy

    def test_both_kinds_missing_get_both_remedies(self) -> None:
        row = check_toolchain({"ruff": None, "pytest": None, "bun": None, "just": None})
        assert "uv sync" in row.remedy and "pacman -S bun just" in row.remedy

    def test_an_empty_string_is_not_a_path(self) -> None:
        assert not check_toolchain({**ALL_TOOLS, "bun": ""}).ok


class TestCt2ComputeType:
    OFFERED = ("float16", "float32", "int8", "int8_float16", "int8_float32")

    def test_wanted_type_offered_passes(self) -> None:
        row = check_ct2_compute_type(self.OFFERED, "int8_float16")
        assert row.ok and "int8_float16" in row.detail

    def test_wanted_type_absent_fails_and_lists_what_cuda_has(self) -> None:
        row = check_ct2_compute_type(["float32", "int8"], "int8_float16")
        assert not row.ok
        assert "float32" in row.detail and "int8" in row.detail
        assert "stt.compute_type" in row.remedy

    def test_library_unavailable_fails_with_the_error(self) -> None:
        row = check_ct2_compute_type(None, "int8_float16", error="ImportError: no ctranslate2")
        assert not row.ok
        assert "ImportError" in row.detail
        assert "uv sync" in row.remedy


class TestGpu:
    def test_enough_free_vram_on_a_capable_gpu_passes(self) -> None:
        row = check_gpu(Gpu(free_mib=5162, total_mib=8151, compute_cap=12.0), need_mib=4062)
        assert row.ok
        assert "5162/8151" in row.detail and "12.0" in row.detail

    def test_too_little_free_vram_fails_and_says_who_to_ask(self) -> None:
        row = check_gpu(Gpu(free_mib=1500, total_mib=8151, compute_cap=12.0), need_mib=4062)
        assert not row.ok
        assert "4062" in row.detail
        assert "nvidia-smi --query-compute-apps" in row.remedy

    def test_an_old_gpu_fails_on_compute_capability_before_vram(self) -> None:
        row = check_gpu(Gpu(free_mib=8000, total_mib=8151, compute_cap=6.1), need_mib=4062)
        assert not row.ok
        assert "6.1" in row.detail
        assert "int8_float16" in row.remedy

    def test_no_nvidia_smi_fails(self) -> None:
        row = check_gpu(None, need_mib=4062)
        assert not row.ok and "nvidia-smi" in row.detail


class TestLlm:
    BASE = "http://127.0.0.1:11434"

    def test_one_word_back_with_thinking_off_passes(self) -> None:
        row = check_llm(LlmProbe("ollama", "neiro-4b", content="Hello"), self.BASE)
        assert row.ok
        assert "ollama" in row.detail and "neiro-4b" in row.detail and "Hello" in row.detail

    def test_nothing_listening_fails_and_names_both_servers(self) -> None:
        row = check_llm(LlmProbe(None, "neiro-4b"), self.BASE)
        assert not row.ok
        assert self.BASE in row.detail
        assert "ollama serve" in row.remedy and "llama-server" in row.remedy

    def test_an_unknown_model_names_the_model_in_the_fix(self) -> None:
        row = check_llm(
            LlmProbe("ollama", "qwen3.5:4b", error="model 'qwen3.5:4b' not found"), self.BASE
        )
        assert not row.ok
        assert "qwen3.5:4b" in row.remedy and "llm.model" in row.remedy

    def test_any_other_error_is_reported_not_swallowed(self) -> None:
        row = check_llm(LlmProbe("llama-server", "m", error="HTTP 500"), self.BASE)
        assert not row.ok and "HTTP 500" in row.detail

    def test_a_reasoning_field_fails_even_with_content_empty(self) -> None:
        # The measured failure: content "" and the budget spent in
        # `message.thinking`, no error anywhere.
        row = check_llm(LlmProbe("ollama", "m", content="", reasoning="Thinking"), self.BASE)
        assert not row.ok
        assert "think: false" in row.remedy and "--reasoning-budget 0" in row.remedy

    def test_an_inline_think_opener_fails(self) -> None:
        row = check_llm(LlmProbe("llama-server", "m", content="<think>\nhmm"), self.BASE)
        assert not row.ok

    def test_a_think_opener_past_the_sniff_window_is_a_reply_not_a_block(self) -> None:
        row = check_llm(LlmProbe("llama-server", "m", content="a" * 64 + "<think>"), self.BASE)
        assert row.ok

    def test_no_content_and_no_known_reasoning_field_still_fails(self) -> None:
        # Some backend will hide reasoning under a name nobody has seen;
        # one token that produced nothing is that bug, whatever it is called.
        row = check_llm(LlmProbe("llama-server", "m", content=""), self.BASE)
        assert not row.ok and "no content" in row.detail


PACTL_SINKS = """Sink #60
\tState: SUSPENDED
\tName: alsa_output.pci-0000_00_1f.3.analog-stereo
\tDescription: Built-in Audio Analog Stereo
\tDriver: PipeWire
\tMute: no
\tVolume: front-left: 19661 /  30% / -31.37 dB
\tProperties:
\t\tnode.name = "alsa_output.pci-0000_00_1f.3.analog-stereo"
Sink #77
\tState: RUNNING
\tName: bluez_output.11_11_22_33_47_91.1
\tDescription: Mivi SuperPods Immersio Pro
\tMute: yes
\tVolume: front-left: 65536 / 100% / 0.00 dB
"""

PACTL_SOURCES = """Source #60
\tName: alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
\tMute: no
Source #61
\tName: alsa_input.pci-0000_00_1f.3.analog-stereo
\tMute: yes
"""


class TestParsePactl:
    def test_ids_names_and_mute_state_come_from_the_blocks(self) -> None:
        nodes = parse_pactl_list("sink", PACTL_SINKS)
        assert [(n.id, n.name, n.muted) for n in nodes] == [
            ("60", "alsa_output.pci-0000_00_1f.3.analog-stereo", False),
            ("77", "bluez_output.11_11_22_33_47_91.1", True),
        ]
        assert all(n.kind == "sink" for n in nodes)

    def test_the_properties_node_name_line_does_not_shadow_name(self) -> None:
        # `node.name = "..."` under Properties is not a `Name:` line.
        assert parse_pactl_list("sink", PACTL_SINKS)[0].name.startswith("alsa_output")

    def test_sources_parse_with_their_own_header(self) -> None:
        nodes = parse_pactl_list("source", PACTL_SOURCES)
        assert [(n.id, n.muted) for n in nodes] == [("60", False), ("61", True)]
        assert nodes[1].kind == "source"

    def test_no_output_is_no_nodes(self) -> None:
        assert parse_pactl_list("sink", "") == []


class TestProfileNode:
    NODES = parse_pactl_list("sink", PACTL_SINKS) + parse_pactl_list("source", PACTL_SOURCES)

    def test_a_live_unmuted_node_passes_with_its_id(self) -> None:
        row = check_profile_node(
            "speakers", "output", "alsa_output.pci-0000_00_1f.3.analog-stereo", self.NODES
        )
        assert row.ok and "#60" in row.detail

    def test_a_muted_node_fails_with_the_exact_wpctl_line(self) -> None:
        row = check_profile_node(
            "earbuds", "output", "bluez_output.11_11_22_33_47_91.1", self.NODES
        )
        assert not row.ok
        assert "MUTED" in row.detail
        assert row.remedy == "wpctl set-mute 77 0"

    def test_a_muted_source_gets_its_own_id_not_the_sinks(self) -> None:
        row = check_profile_node(
            "speakers", "input", "alsa_input.pci-0000_00_1f.3.analog-stereo", self.NODES
        )
        assert row.remedy == "wpctl set-mute 61 0"

    def test_an_absent_node_fails_and_points_at_the_profile_switch(self) -> None:
        row = check_profile_node("earbuds", "input", "bluez_input.11:11:22:33:47:91", self.NODES)
        assert not row.ok
        assert "not a live source" in row.detail
        assert "active_profile" in row.remedy

    def test_kind_is_honoured_a_sink_name_is_not_an_input(self) -> None:
        row = check_profile_node(
            "speakers", "input", "alsa_output.pci-0000_00_1f.3.analog-stereo", self.NODES
        )
        assert not row.ok

    def test_a_profile_with_no_device_named_fails(self) -> None:
        row = check_profile_node("earbuds", "input", None, self.NODES)
        assert not row.ok and "input_device" in row.detail

    def test_no_active_profile_fails_and_names_the_inventory(self) -> None:
        row = check_profile_node("", "output", None, self.NODES)
        assert not row.ok and "audio-inventory" in row.remedy


class TestSamplerates:
    def test_both_rates_accepted_passes(self) -> None:
        row = check_samplerates(16000, 24000, None, None, via="pulse → speakers")
        assert row.ok and "speakers" in row.detail

    def test_a_refused_input_rate_fails_naming_the_rate(self) -> None:
        row = check_samplerates(16000, 24000, "Invalid sample rate", None, via="pulse")
        assert not row.ok
        assert "input 16000" in row.detail
        assert "PipeWire resamples" in row.remedy

    def test_a_refused_output_rate_fails_naming_the_rate(self) -> None:
        row = check_samplerates(16000, 24000, None, "Invalid sample rate", via="pulse")
        assert not row.ok and "output 24000" in row.detail

    def test_no_pulse_device_at_all_points_at_pipewire_pulse(self) -> None:
        row = check_samplerates(16000, 24000, "No input device matching 'pulse'", None, via="pulse")
        assert not row.ok and "pipewire-pulse" in row.remedy


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_model(
    root: Path,
    name: str,
    files: dict[str, bytes],
    lfs: set[str] = frozenset(),
    marker: bool = True,
) -> Path:
    """A downloaded model as huggingface_hub leaves it in a local_dir:
    the files, a `.cache/huggingface/download/<file>.metadata` receipt
    per file (commit, etag, timestamp), and fetch_models.py's marker.
    LFS receipts carry the sha256; the others a 40-hex git blob id.
    """
    model_dir = root / name
    cache = model_dir / ".cache" / "huggingface" / "download"
    for relative, data in files.items():
        target = model_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        etag = _sha256(data) if relative in lfs else hashlib.sha1(data).hexdigest()
        meta = cache / (relative + ".metadata")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(f"{'0' * 40}\n{etag}\n1789256574.1\n")
    if marker:
        (model_dir / doctor.MODEL_COMPLETE_MARKER).write_text("{}")
    return model_dir


class TestModelFiles:
    def test_present_and_matching_passes(self, tmp_path: Path) -> None:
        _make_model(
            tmp_path, "m", {"model.bin": b"weights", "config.json": b"{}"}, lfs={"model.bin"}
        )
        row = check_model_files(["m"], tmp_path)
        assert row.ok and "1/1" in row.detail

    def test_a_corrupt_lfs_file_fails_by_hash_not_presence(self, tmp_path: Path) -> None:
        model = _make_model(tmp_path, "m", {"model.bin": b"weights"}, lfs={"model.bin"})
        (model / "model.bin").write_bytes(b"weightx")  # same length, wrong bytes
        row = check_model_files(["m"], tmp_path)
        assert not row.ok
        assert "model.bin sha256 mismatch" in row.detail
        assert "--only m --force" in row.remedy

    def test_a_missing_file_fails_even_with_the_marker_present(self, tmp_path: Path) -> None:
        model = _make_model(tmp_path, "m", {"model.bin": b"weights"}, lfs={"model.bin"})
        (model / "model.bin").unlink()
        row = check_model_files(["m"], tmp_path)
        assert not row.ok and "model.bin missing" in row.detail

    def test_not_downloaded_names_the_fetch_command(self, tmp_path: Path) -> None:
        row = check_model_files(["never-fetched"], tmp_path)
        assert not row.ok
        assert "not downloaded" in row.detail
        assert "fetch-models --runs local --purpose runtime" in row.remedy

    def test_a_marker_without_receipts_is_not_verified(self, tmp_path: Path) -> None:
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / doctor.MODEL_COMPLETE_MARKER).write_text("{}")
        assert verify_model_dir(tmp_path / "m") == ["no download metadata to verify against"]

    def test_a_non_lfs_file_is_presence_only(self, tmp_path: Path) -> None:
        # Its receipt is a git blob id, not a content hash; changing the
        # bytes must not read as corruption.
        model = _make_model(tmp_path, "m", {"README.md": b"hello"})
        (model / "README.md").write_bytes(b"edited")
        assert verify_model_dir(model) == []

    def test_receipts_map_nested_paths(self, tmp_path: Path) -> None:
        data = b"onnx"
        model = _make_model(tmp_path, "m", {"onnx/model.onnx": data}, lfs={"onnx/model.onnx"})
        assert expected_hashes(model) == {"onnx/model.onnx": _sha256(data)}

    def test_the_digest_is_what_decides(self, tmp_path: Path) -> None:
        # A verifier that only looked at sizes would pass this.
        _make_model(tmp_path, "m", {"model.bin": b"weights"}, lfs={"model.bin"})
        row = check_model_files(["m"], tmp_path, digest=lambda path: "0" * 64)
        assert not row.ok

    def test_several_bad_models_are_counted_not_hidden(self, tmp_path: Path) -> None:
        _make_model(tmp_path, "good", {"a": b"1"}, lfs={"a"})
        row = check_model_files(["good", "x", "y"], tmp_path)
        assert not row.ok
        assert "1/3" in row.detail and "+1 more" in row.detail


class TestNoVoiceInGit:
    def test_nothing_tracked_is_clean(self) -> None:
        assert check_no_voice_in_git([]).ok

    def test_a_synthetic_fixture_is_allowed(self) -> None:
        assert check_no_voice_in_git(["scripts/fixtures/tone_3s.wav"]).ok

    def test_anything_under_data_voice_fails(self) -> None:
        row = check_no_voice_in_git(["data/voice/yash/001.wav"])
        assert not row.ok and "git rm --cached" in row.remedy

    def test_audio_outside_the_fixture_roots_fails(self) -> None:
        row = check_no_voice_in_git(["scripts/fixtures/tone_3s.wav", "tests/me.flac"])
        assert not row.ok and "tests/me.flac" in row.detail


class TestWsAuth:
    def test_refused_without_and_hello_with_passes(self) -> None:
        row = check_ws_auth(WsOutcome(False, 1008, True))
        assert row.ok and "1008" in row.detail

    def test_an_accepted_anonymous_peer_fails(self) -> None:
        row = check_ws_auth(WsOutcome(True, None, True))
        assert not row.ok and "accept()" in row.remedy

    def test_refusing_everyone_is_not_a_pass(self) -> None:
        # The PEP 563 regression looked exactly like this.
        row = check_ws_auth(WsOutcome(False, 1008, False))
        assert not row.ok and "PEP 563" in row.remedy

    def test_a_probe_that_could_not_run_fails(self) -> None:
        row = check_ws_auth(WsOutcome(False, None, False, error="ImportError: starlette"))
        assert not row.ok and "ImportError" in row.detail

    def test_the_real_app_in_process_passes_the_row(self) -> None:
        # No server, no port: the probe drives build_app through
        # Starlette's client, which is how the doctor runs it too.
        outcome = doctor._observe_ws()
        assert outcome.error == ""
        assert check_ws_auth(outcome).ok


class TestGuard:
    def test_a_row_that_throws_becomes_a_red_row_not_a_traceback(self) -> None:
        def boom() -> Check:
            raise RuntimeError("pactl exploded")

        row = doctor._row("profile output node", boom)
        assert not row.ok
        assert row.name == "profile output node"
        assert "RuntimeError" in row.detail and "pactl exploded" in row.detail

    def test_a_row_that_returns_is_passed_through(self) -> None:
        good = Check("x", True, "fine")
        assert doctor._row("x", lambda: good) is good


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(doctor, "console", Console(file=buffer, width=200, force_terminal=False))
    return buffer


class TestRender:
    def test_all_green_is_exit_0_and_says_pass(self, captured: io.StringIO) -> None:
        code = render([Check("a", True, "ok"), Check("b", True, "ok")])
        out = captured.getvalue()
        assert code == 0
        assert "PASS all 2" in out
        assert "FAIL" not in out

    def test_one_red_row_makes_the_summary_red(self, captured: io.StringIO) -> None:
        code = render(
            [Check("a", True, "ok"), Check("b", False, "muted", remedy="wpctl set-mute 77 0")]
        )
        out = captured.getvalue()
        assert code == 1
        assert "FAIL 1 of 2" in out

    def test_a_red_row_carries_its_fix_on_its_own_line(self, captured: io.StringIO) -> None:
        render([Check("b", False, "muted", remedy="wpctl set-mute 77 0")])
        line = next(line for line in captured.getvalue().splitlines() if "muted" in line)
        assert "wpctl set-mute 77 0" in line


class TestRunDoctor:
    def test_exit_code_follows_the_rows(self, captured: io.StringIO) -> None:
        red = [Check("a", True, "ok"), Check("b", False, "no", remedy="fix")]
        green = [Check("a", True, "ok")]
        assert run_doctor(gather=lambda: red) == 1
        assert run_doctor(gather=lambda: green) == 0

    def test_audio_inventory_only_short_circuits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "audio_inventory", lambda: 7)
        touched = []
        assert run_doctor(audio_inventory_only=True, gather=lambda: touched.append(1) or []) == 7
        assert touched == []
