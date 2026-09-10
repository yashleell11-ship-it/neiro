"""The `neiro` command. Stage 0 wires up `doctor`; later stages add
`run`, `bench`, `wer`, `record-set`, `say`, `chat`, `mic-test`, `pause`,
`fetch-models`, `licences`, `uninstall` — see the plan's task list.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False, help="Neiro — an anime character who lives on your machine."
)


@app.callback()
def _main() -> None:
    """Neiro — an anime character who lives on your machine.

    A no-op callback. Without it, Typer collapses a single-command app so
    `neiro doctor` fails with "unexpected extra argument (doctor)" — this
    keeps `doctor` (and every command added in later stages) addressable
    by name.
    """


@app.command()
def doctor(
    audio_inventory: bool = typer.Option(
        False, "--audio-inventory", help="Stage 0 Task 0: enumerate audio devices."
    ),
) -> None:
    """Check every assumption the project makes, with a fix for each failure."""
    from neiro.doctor import run_doctor

    raise typer.Exit(code=run_doctor(audio_inventory_only=audio_inventory))


@app.command(name="mic-test")
def mic_test(
    duration: float = typer.Option(3.0, "--duration", help="Seconds to record."),
    profile: str = typer.Option(
        None,
        "--profile",
        help="Test a specific profile (e.g. 'speakers') without changing the "
        "saved active_profile in config.toml. Useful when the usual profile "
        "isn't reachable right now (Bluetooth off, etc.) — see Gate G4.",
    ),
) -> None:
    """Stage 0 Task 3: record from the active audio profile, then play it
    back twice — once at the correct 16 kHz, once deliberately at 24 kHz.

    The second playback is meant to sound wrong (sped-up, higher-pitched
    — the "chipmunk" effect). That is the point: it is what a
    sample-rate mismatch sounds like, so every future "audio sounds
    wrong" bug becomes something you recognise by ear in three seconds
    instead of something you have to debug from scratch.
    """
    import wave

    from rich.console import Console

    from neiro.audio.capture import record
    from neiro.audio.playback import play
    from neiro.config import Neiro

    console = Console()
    cfg = Neiro()
    if profile is not None:
        if profile not in cfg.audio.profiles:
            console.print(
                f"[red]No such profile '{profile}'.[/red] Known: {list(cfg.audio.profiles)}"
            )
            raise typer.Exit(code=1)
        cfg = cfg.model_copy(
            update={"audio": cfg.audio.model_copy(update={"active_profile": profile})}
        )

    console.print(
        f"[bold]Recording {duration}s from profile '{cfg.audio.active_profile}'...[/bold]"
    )
    audio = record(duration, cfg=cfg)

    import numpy as np

    peak = float(np.abs(audio).max())
    console.print(f"Captured {audio.shape[0]} samples. Peak level: {peak:.4f}")
    if peak < 1e-4:
        console.print(
            "[yellow]That's silent.[/yellow] Check the active profile's input device "
            "isn't muted (neiro doctor) and that it's the device you're actually "
            "speaking into."
        )

    out_path = "/tmp/neiro-mic-test.wav"
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(cfg.audio.input_samplerate)
        pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        w.writeframes(pcm16.tobytes())
    console.print(f"Saved to {out_path}")

    console.print("\n[bold]Playing back at 16000 Hz (correct)...[/bold]")
    play(audio, samplerate=cfg.audio.input_samplerate, cfg=cfg)

    console.print("[bold]Playing back at 24000 Hz (deliberately wrong — chipmunk)...[/bold]")
    play(audio, samplerate=24000, cfg=cfg)

    console.print(
        "\n[green]Done.[/green] First one should have sounded like you. Second one "
        "should have sounded sped-up and higher-pitched — that's a rate mismatch, "
        "and now you know what it sounds like."
    )


@app.command()
def ptt() -> None:
    """Stage 0 Task 4: push-to-talk in the terminal. SPACE to start/stop
    recording, q to quit — writes each recording to /tmp/neiro-last.wav.

    The system-wide version (a Hyprland keybind, no terminal needed) is
    Task 11. This one exists so the loop can be proven before adding a
    compositor bind, a socket, and a compiled helper on top of it.
    """
    from neiro.audio.ptt import run as run_ptt
    from neiro.config import Neiro

    raise typer.Exit(code=run_ptt(Neiro()))


if __name__ == "__main__":
    app()
