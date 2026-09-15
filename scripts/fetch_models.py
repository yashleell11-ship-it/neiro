#!/usr/bin/env python3
"""Pull the models in src/elizabeth/modelspec.py.

    uv run scripts/fetch_models.py --dry-run           # the plan, with REAL sizes re-read from HF
    uv run scripts/fetch_models.py --runs local        # what the laptop/hostel tier needs
    uv run scripts/fetch_models.py --runs box          # what goes to the 3090 Ti
    uv run scripts/fetch_models.py --purpose training  # trainable safetensors only
    uv run scripts/fetch_models.py --only qwen3.5-4b-gguf

Resumable — HF resumes partial files, and a finished model gets a
`.elizabeth-complete` marker so a re-run skips it. `--dry-run` asks the Hub
for the real byte count of exactly the files that would be fetched,
because modelspec's numbers are a record, not a promise.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from elizabeth.modelspec import ModelSpec, select, total_gb

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "models"
# Where the page looks for a VRM (`avatar_url` in src/elizabeth/server.py).
# An avatar is served to the browser, not loaded by the daemon, so it
# lives with the page — gitignored there, see .gitignore — not under
# models/ with the weights.
AVATAR_DEST = REPO / "web" / "public" / "avatar"
MARKER = ".elizabeth-complete"

console = Console()


def real_size_gb(spec: ModelSpec) -> float | None:
    """Byte count of exactly the files this spec would pull. None if the
    Hub can't be reached — a size we can't check is not a size we report.
    """
    from huggingface_hub import get_token, list_repo_tree

    try:
        total = 0
        for entry in list_repo_tree(
            spec.hf_id, revision=spec.revision, recursive=True, token=get_token()
        ):
            path = getattr(entry, "path", "")
            size = (
                getattr(entry, "size", None) or getattr(getattr(entry, "lfs", None), "size", 0) or 0
            )
            if spec.allow and not any(fnmatch.fnmatch(path, pat) for pat in spec.allow):
                continue
            total += size
        return round(total / 1e9, 2)
    except Exception:  # noqa: BLE001 — an unreachable Hub is a "?" in the table, not a crash
        return None


def dest_for(spec: ModelSpec, dest: Path = DEFAULT_DEST) -> Path:
    """`<dest>/<name>` — except an avatar, which goes where the page looks
    for it, or `elizabeth fetch-models --component avatar` would download a
    file the face never finds.
    """
    if spec.component == "avatar":
        return AVATAR_DEST / spec.name
    return dest / spec.name


def fetch(spec: ModelSpec, dest: Path, force: bool) -> str:
    from huggingface_hub import get_token, snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    marker = dest / MARKER
    if marker.exists() and not force:
        return "already"
    try:
        snapshot_download(
            repo_id=spec.hf_id,
            revision=spec.revision,
            local_dir=dest,
            allow_patterns=spec.allow or None,
            token=get_token(),
            max_workers=8,
        )
    except RepositoryNotFoundError:
        return "not-found"
    except GatedRepoError:
        return "needs-approval"
    except KeyboardInterrupt:
        return "interrupted"
    except Exception as exc:  # noqa: BLE001 — the status is the message
        console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        return "failed"
    dest.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(spec.model_dump(), indent=1))
    return "done"


REMEDY = {
    "needs-approval": "accept the model's terms in a browser, then re-run",
    "not-found": "the hf_id in modelspec.py is wrong — fix it",
    "failed": "re-run; downloads resume",
    "interrupted": "re-run; downloads resume",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan with real sizes, download nothing"
    )
    ap.add_argument("--purpose", choices=["runtime", "training"])
    ap.add_argument(
        "--runs", choices=["local", "box"], help="which machine needs it ('both' always included)"
    )
    ap.add_argument("--component", choices=["llm", "stt", "tts", "vad", "turn", "ser", "avatar"])
    ap.add_argument("--only", action="append", help="model name(s) from modelspec.py; repeatable")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    items = select(purpose=args.purpose, runs=args.runs, component=args.component, only=args.only)
    if not items:
        console.print("[yellow]nothing selected[/]")
        return 0

    table = Table(title=f"{len(items)} models, {total_gb(items)} GB per modelspec")
    for col in ("name", "component", "purpose", "runs", "recorded", "actual", "licence"):
        table.add_column(col)
    checked = 0.0
    unknown = False
    for m in items:
        actual = real_size_gb(m) if args.dry_run else None
        if actual is None:
            if args.dry_run:
                unknown = True
            shown = "?" if args.dry_run else "-"
            checked += m.size_gb
        else:
            drift = abs(actual - m.size_gb) / max(m.size_gb, 0.01)
            shown = f"[red]{actual}[/]" if drift > 0.3 else str(actual)
            checked += actual
        table.add_row(m.name, m.component, m.purpose, m.runs, f"{m.size_gb:.2f}", shown, m.license)
    console.print(table)

    free_gb = shutil.disk_usage(args.dest if args.dest.exists() else REPO).free / 1e9
    console.print(f"to download: [bold]{checked:.1f} GB[/]   free on disk: {free_gb:.0f} GB")
    if unknown:
        console.print("[yellow]'?' = the Hub could not be reached for that repo[/]")
    if checked > free_gb * 0.85:
        console.print(
            "[red]that does not fit comfortably[/] — narrow with --runs / --component / --only"
        )
        return 2
    if args.dry_run:
        return 0

    results: dict[str, str] = {}
    for m in items:
        console.rule(f"{m.name}  ({m.size_gb:.2f} GB)  {m.hf_id}")
        results[m.name] = fetch(m, dest_for(m, args.dest), args.force)
        console.print(f"→ {results[m.name]}")

    summary = Table(title="summary")
    for col in ("model", "status", "what to do"):
        summary.add_column(col)
    bad = 0
    for name, status in results.items():
        if status not in ("done", "already"):
            bad += 1
        summary.add_row(name, status, REMEDY.get(status, ""))
    console.print(summary)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
