#!/usr/bin/env python3
"""Pull Neiro's training datasets per data/datasets.toml.

    uv run scripts/fetch_datasets.py --dry-run            # what tier-1 would pull, and how much
    uv run scripts/fetch_datasets.py                      # pull tier 1 (essentials)
    uv run scripts/fetch_datasets.py --tier 2             # tier 1 + 2
    uv run scripts/fetch_datasets.py --target tts_voice   # one component
    uv run scripts/fetch_datasets.py --only svarah --only crema_d

Resumable: HF downloads resume natively, URL downloads use aria2c -c /
wget -c. A finished dataset gets a `.neiro-complete` marker and is
skipped next time (unless --force). Gated datasets are never fetched
silently — the script prints the exact URL where the terms must be
accepted, because that acceptance is a licence decision a person makes.

Never commits anything; data/datasets/ is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from neiro.training.manifest import TARGETS, Dataset, Manifest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "datasets.toml"
DEFAULT_DEST = REPO / "data" / "datasets"
MARKER = ".neiro-complete"

console = Console()


def _hf_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def fetch_hf(ds: Dataset, dest: Path, token: str | None) -> str:
    from huggingface_hub import dataset_info, snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        info = dataset_info(ds.hf_id, token=token)
    except RepositoryNotFoundError:
        return "not-found"
    if info.gated and not token:
        return "needs-login"
    try:
        snapshot_download(
            repo_id=ds.hf_id,
            repo_type="dataset",
            local_dir=dest,
            token=token,
            max_workers=8,
        )
    except GatedRepoError:
        return "needs-approval"
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        # One corpus must never end a 100 GB run. Happened for real:
        # deleting a directory while its download was in flight raised
        # FileNotFoundError out of hf_hub's move step and killed every
        # remaining dataset in the queue. Everything here resumes, so the
        # right response is to record it and carry on.
        console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        return "failed"
    return "done"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(ds: Dataset, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        cmd = ["aria2c", "-c", "-x", "4", "-s", "4", "--dir", str(dest), ds.url]
    else:
        cmd = ["wget", "-c", "-P", str(dest), ds.url]
    try:
        if subprocess.run(cmd, check=False).returncode != 0:
            return "failed"
    except KeyboardInterrupt:
        raise
    except OSError as exc:
        console.print(f"[red]{type(exc).__name__}[/]: {exc}")
        return "failed"
    if ds.sha256:
        target = dest / ds.url.rsplit("/", 1)[-1]
        if not target.exists() or _sha256(target) != ds.sha256:
            return "checksum-mismatch"
    return "done"


# Access paths a script can complete on its own. Everything else needs a
# person to agree to something.
AUTOMATABLE = ("direct", "hf-login")


def fetch(ds: Dataset, dest: Path, token: str | None, force: bool) -> str:
    marker = dest / MARKER
    if marker.exists() and not force:
        return "already"
    if ds.access not in AUTOMATABLE:
        # Do not even try. A request-form corpus whose `url` is its
        # landing page will download that HTML page, exit 0, and get
        # marked complete -- which is how you end up with 52 KB standing
        # in for 47 GB and a training run that silently has no data.
        # Observed exactly that with MSP-Podcast on 2026-09-13.
        return f"needs-{ds.access}"
    status = fetch_hf(ds, dest, token) if ds.is_hf else fetch_url(ds, dest)
    if status == "done":
        dest.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(ds.model_dump(), indent=1))
    return status


REMEDY = {
    "needs-request-form": "apply on the dataset's own site, then download by hand into data/datasets/<name>/",
    "needs-paid": "this one costs money — decide before spending anything",
    "needs-unavailable": "no working download path was found; the manifest entry records why",
    "needs-login": "run `uv run hf auth login` (a read token from huggingface.co/settings/tokens)",
    "needs-approval": "accept the terms in your browser, then re-run",
    "not-found": "the hf_id in data/datasets.toml is wrong — fix the manifest",
    "failed": "download failed — re-run, it resumes",
    "checksum-mismatch": "file differs from the manifest's sha256 — delete it and re-run",
}


def plan_table(items: list[Dataset], title: str) -> Table:
    t = Table(title=title, show_lines=False)
    for col in ("name", "target", "p", "GB", "hours", "licence", "access", "publish"):
        t.add_column(col)
    for d in items:
        t.add_row(
            d.name,
            d.target,
            str(d.priority),
            f"{d.size_gb:.1f}",
            f"{d.hours:.0f}" if d.hours else "-",
            d.license,
            d.access,
            d.weights_publishable,
        )
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan and totals, download nothing"
    )
    ap.add_argument(
        "--tier",
        type=int,
        default=1,
        help="priority ceiling: 1 essentials (default), 2, or 3 = everything",
    )
    ap.add_argument("--target", choices=TARGETS, help="restrict to one training target")
    ap.add_argument("--only", action="append", help="dataset name(s) from the manifest; repeatable")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true", help="re-fetch even if marked complete")
    args = ap.parse_args(argv)

    manifest = Manifest.load(MANIFEST)
    items = manifest.select(max_priority=args.tier, target=args.target, only=args.only)
    if not items:
        console.print(
            "[yellow]nothing selected[/] — the manifest may be empty or the filter too narrow"
        )
        return 0

    console.print(
        plan_table(
            items,
            f"{len(items)} datasets, {Manifest.total_gb(items)} GB, {Manifest.total_hours(items)} h audio",
        )
    )
    free_gb = shutil.disk_usage(args.dest.parent if args.dest.exists() else REPO).free / 1e9
    console.print(f"free on disk: {free_gb:.0f} GB")
    if Manifest.total_gb(items) > free_gb * 0.8:
        console.print("[red]that does not fit comfortably — narrow the selection[/]")
        if not args.dry_run:
            return 2
    if args.dry_run:
        return 0

    token = _hf_token()
    if token is None:
        console.print(
            "[yellow]no HF token[/] — ungated datasets still download; gated ones will be listed at the end"
        )

    results: dict[str, str] = {}
    for d in items:
        console.rule(f"{d.name} ({d.size_gb:.1f} GB)")
        results[d.name] = fetch(d, args.dest / d.name, token, args.force)
        console.print(f"→ {results[d.name]}")

    summary = Table(title="summary")
    summary.add_column("dataset")
    summary.add_column("status")
    summary.add_column("what to do")
    bad = 0
    for name, status in results.items():
        remedy = REMEDY.get(status, "")
        ds = next(x for x in items if x.name == name)
        if status.startswith("needs-") and ds.access not in AUTOMATABLE:
            # Expected, not a failure: the manifest said so up front.
            remedy = f"{remedy} — {ds.source}"
        elif status not in ("done", "already"):
            bad += 1
            if status == "needs-approval":
                remedy = f"accept at {ds.source}, then re-run"
        summary.add_row(name, status, remedy)
    console.print(summary)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
