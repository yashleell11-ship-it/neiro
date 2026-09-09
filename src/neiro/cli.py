"""The `neiro` command. Stage 0 wires up `doctor`; later stages add
`run`, `bench`, `wer`, `record-set`, `say`, `chat`, `mic-test`, `pause`,
`fetch-models`, `licences`, `uninstall` — see the plan's task list.
"""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="Neiro — an anime character who lives on your machine.")


@app.command()
def doctor(
    audio_inventory: bool = typer.Option(
        False, "--audio-inventory", help="Stage 0 Task 0: enumerate audio devices."
    ),
) -> None:
    """Check every assumption the project makes, with a fix for each failure."""
    from neiro.doctor import run_doctor

    raise typer.Exit(code=run_doctor(audio_inventory_only=audio_inventory))


if __name__ == "__main__":
    app()
