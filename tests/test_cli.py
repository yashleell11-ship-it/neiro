"""Tests for the `neiro` command surface (src/neiro/cli.py).

Nothing here loads a model. cli.py imports only typer at module level
and every command imports its own dependencies inside its body, so
importing `neiro.cli` costs nothing -- and these tests keep it that way
by stubbing the one heavy import a command makes rather than paying for
it.
"""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest
import typer
from typer.main import get_command

from neiro import cli


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
        # The set `neiro --help` shows is the set of decorators that ran.
        # One pasted under a different Typer, or under something that
        # exits the process, reads as present in the file and is absent
        # from the command line.
        assert _commands_in_source()
        assert set(get_command(cli.app).commands) == _commands_in_source()

    def test_running_the_module_directly_registers_every_command(self, monkeypatch) -> None:
        # `python -m neiro.cli`. The console script imports the module,
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
