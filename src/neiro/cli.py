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
def stt(
    wav_path: str = typer.Argument(..., help="WAV file to transcribe, e.g. /tmp/neiro-last.wav"),
) -> None:
    """Stage 0 Task 5: transcribe a WAV file with faster-whisper, print
    the result and how long it took.

    Composes with `neiro ptt`: record with ptt, transcribe the result
    with `neiro stt /tmp/neiro-last.wav`.
    """
    import asyncio
    import time

    import soundfile as sf
    from rich.console import Console

    from neiro.config import Neiro
    from neiro.stt.faster_whisper import FasterWhisperStt

    console = Console()
    cfg = Neiro()

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != cfg.audio.input_samplerate:
        console.print(
            f"[yellow]Warning:[/yellow] '{wav_path}' is {sr} Hz, STT expects "
            f"{cfg.audio.input_samplerate} Hz — no resampling is applied here, "
            "the transcript may be affected."
        )

    engine = FasterWhisperStt(cfg)
    try:
        t0 = time.perf_counter()
        text = asyncio.run(engine.transcribe(audio))
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if text:
        console.print(f"[bold]{text}[/bold]")
    else:
        console.print(
            "[yellow]didn't catch that[/yellow] "
            "(silence, low confidence, or a known hallucination — see stt/faster_whisper.py)"
        )
    console.print(f"({elapsed_ms:.0f} ms)")


@app.command(name="record-set")
def record_set(
    name: str = typer.Option("wer", "--name", help="Dataset name, e.g. 'wer' or 'emotion'."),
    count: int = typer.Option(30, "--count", help="How many takes to record in total."),
) -> None:
    """Stage 0 Task 6: record an evaluation set in your own voice.

    Prompts for the reference transcript before each take, so it's
    captured while you still remember what you said. Resumes where it
    left off if you stop partway. Raw WAVs are gitignored (voice is
    biometric); only refs.jsonl is committed.
    """
    from neiro.config import Neiro
    from neiro.recordset import run as run_record_set

    raise typer.Exit(code=run_record_set(name=name, count=count, cfg=Neiro()))


@app.command()
def wer(
    name: str = typer.Option("wer", "--name", help="Dataset name under data/voice/."),
) -> None:
    """Stage 0 Gate G3a: score the current STT against your own voice.

    Every STT swap for the life of this project gets scored against this
    same set — it is the only ground truth that will ever exist for how
    the model performs on *your* voice, in *your* room, on *your* mic.
    """
    import asyncio
    import time
    from pathlib import Path

    import soundfile as sf
    from rich.console import Console
    from rich.table import Table

    from neiro.config import Neiro
    from neiro.evals.wer import load_references, score
    from neiro.recordset import DATA_ROOT
    from neiro.stt.faster_whisper import FasterWhisperStt

    console = Console()
    cfg = Neiro()

    refs_path = Path(DATA_ROOT) / name / "refs.jsonl"
    if not refs_path.exists():
        console.print(
            f"[red]No dataset at {refs_path}[/red]\n"
            f"Record one first:  neiro record-set --name {name} --count 30"
        )
        raise typer.Exit(code=1)

    entries = load_references(refs_path)
    console.print(f"Scoring [bold]{len(entries)}[/bold] utterances from {refs_path}\n")

    engine = FasterWhisperStt(cfg)
    try:
        warm_s = engine.warm()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"(model warmed in {warm_s:.1f}s)\n")

    pairs = []
    total_ms = 0.0
    for entry in entries:
        wav_path = refs_path.parent / entry["file"]
        if not wav_path.exists():
            console.print(f"[yellow]missing {wav_path}, skipping[/yellow]")
            continue
        audio, file_sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if file_sr != cfg.audio.input_samplerate:
            # Would silently produce garbage transcripts and a garbage
            # WER — loud is better than a quietly wrong number.
            console.print(
                f"[red]{entry['file']} is {file_sr} Hz, expected "
                f"{cfg.audio.input_samplerate} Hz — skipping rather than "
                "reporting a wrong score.[/red]"
            )
            continue
        t0 = time.perf_counter()
        hypothesis = asyncio.run(engine.transcribe(audio))
        total_ms += (time.perf_counter() - t0) * 1000
        pairs.append((entry["file"], entry["reference"], hypothesis))

    if not pairs:
        console.print("[red]No audio files found to score.[/red]")
        raise typer.Exit(code=1)

    result = score(pairs)

    table = Table(title=f"WER — {cfg.stt.model_id}")
    table.add_column("file")
    table.add_column("WER", justify="right")
    table.add_column("reference")
    table.add_column("heard")
    for u in result.utterances:
        colour = "green" if u.wer == 0 else ("yellow" if u.wer < 0.3 else "red")
        heard = u.hypothesis or "[dim](rejected)[/dim]"
        table.add_row(u.name, f"[{colour}]{u.wer:.1%}[/{colour}]", u.reference, heard)
    console.print(table)

    console.print(
        f"\n[bold]Corpus WER: {result.wer:.2%}[/bold] "
        f"({result.total_errors} errors / {result.total_ref_words} reference words)"
    )
    console.print(f"Mean transcribe time: {total_ms / len(pairs):.0f} ms per utterance")
    console.print(
        "\nUnder ~10% is fine. Over ~15% means mic gain or room, not the model — "
        "check the peak levels in refs.jsonl before blaming the STT."
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
