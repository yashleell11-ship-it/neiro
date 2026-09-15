"""Tests for the `elizabeth` command surface (src/elizabeth/cli.py).

Nothing here loads a model. cli.py imports only typer at module level
and every command imports its own dependencies inside its body, so
importing `elizabeth.cli` costs nothing -- and these tests keep it that way
by stubbing the one heavy import a command makes rather than paying for
it.
"""

from __future__ import annotations

import ast
import json
import runpy
import sys
import time
import types
import wave
from pathlib import Path

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

from elizabeth import cli


def _commands_in_source() -> set[str]:
    """Every `@app.command(...)` in cli.py, by the name Typer gives it.

    Read from the file rather than the imported module, because the
    imported module is the thing under test: a decorator that never ran
    is invisible to introspection and perfectly visible in the source.
    """
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
                continue
            target = deco.func.value
            if not (isinstance(target, ast.Name) and target.id == "app"):
                continue
            if deco.func.attr != "command":
                continue
            explicit = next((kw.value for kw in deco.keywords if kw.arg == "name"), None)
            if isinstance(explicit, ast.Constant):
                names.add(explicit.value)
            else:
                names.add(node.name.replace("_", "-"))
    return names


class TestEveryCommandIsRegistered:
    def test_the_source_and_the_app_agree(self) -> None:
        # The set `elizabeth --help` shows is the set of decorators that ran.
        # One pasted under a different Typer, or under something that
        # exits the process, reads as present in the file and is absent
        # from the command line.
        assert _commands_in_source()
        assert set(get_command(cli.app).commands) == _commands_in_source()

    def test_running_the_module_directly_registers_every_command(self, monkeypatch) -> None:
        # `python -m elizabeth.cli`. The console script imports the module,
        # so every decorator runs before anything is parsed; the direct
        # path calls `app()` at whatever line the __main__ guard sits on,
        # and a command defined below that line does not exist. For a
        # while that was four of them.
        seen: list[typer.Typer] = []

        def capture(self: typer.Typer, *args: object, **kwargs: object) -> None:
            # What the real `app()` does that matters here: it never
            # returns, so nothing after the guard is evaluated.
            seen.append(self)
            raise SystemExit(0)

        monkeypatch.setattr(typer.Typer, "__call__", capture)
        with pytest.raises(SystemExit):
            runpy.run_path(cli.__file__, run_name="__main__")

        assert len(seen) == 1
        assert set(get_command(seen[0]).commands) == _commands_in_source()


class TestWerReportsPercentiles:
    """`elizabeth wer` is the command every STT swap is scored with, so its
    transcribe time is the one place a mean would do the most damage.
    """

    # Three quick utterances and one that took four seconds -- a long
    # clip, a cold cache, a cuBLAS hiccup. Nearest-rank p50 is 100 and
    # p95 is 4000; the mean is 1075, a number nobody experienced.
    TRANSCRIBE_S = (0.1, 0.1, 0.1, 4.0)

    def test_transcribe_time_is_p50_and_p95_never_a_mean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from elizabeth import recordset
        from elizabeth.config import Elizabeth

        # A dataset the command will accept: refs.jsonl plus silent WAVs
        # at the configured input rate, so nothing is skipped as a rate
        # mismatch.
        samplerate = Elizabeth().audio.input_samplerate
        dataset = tmp_path / "wer"
        dataset.mkdir()
        with (dataset / "refs.jsonl").open("w") as refs:
            for i, _ in enumerate(self.TRANSCRIBE_S):
                wav_name = f"{i + 1:03d}.wav"
                with wave.open(str(dataset / wav_name), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(samplerate)
                    w.writeframes(bytes(2 * samplerate))
                refs.write(json.dumps({"file": wav_name, "reference": "hello"}) + "\n")
        monkeypatch.setattr(recordset, "DATA_ROOT", tmp_path)

        # The clock only moves inside transcribe(), by a scripted amount,
        # so the measured times are exact whatever else reads it.
        clock = {"now": 0.0}
        durations = iter(self.TRANSCRIBE_S)
        monkeypatch.setattr(time, "perf_counter", lambda: clock["now"])

        class FakeStt:
            def __init__(self, cfg: object) -> None:
                pass

            def warm(self) -> float:
                return 0.0

            async def transcribe(self, audio: object) -> str:
                clock["now"] += next(durations)
                return "hello"

        # Stubbed at the module level: the real one imports faster-whisper
        # and loads a model, and the output format is what is under test.
        stub = types.ModuleType("elizabeth.stt.faster_whisper")
        stub.FasterWhisperStt = FakeStt  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "elizabeth.stt.faster_whisper", stub)

        result = CliRunner().invoke(cli.app, ["wer", "--name", "wer"])

        assert result.exit_code == 0, result.output
        assert "p50 100 ms" in result.output
        assert "p95 4000 ms" in result.output
        assert "Mean" not in result.output
        assert "1075" not in result.output
