#!/usr/bin/env python3
"""Prepare the TEXT corpora for the persona / Hinglish / tool-calling LoRA.

    cd training && uv run python ../scripts/prep_text.py                 # every complete download
    cd training && uv run python ../scripts/prep_text.py --only goemotions --limit 200
    cd training && uv run python ../scripts/prep_text.py --publishable-only
    cd training && uv run python ../scripts/prep_text.py --dry-run       # the plan, nothing written

Reads data/datasets/<name>/ for every text corpus in data/datasets.toml
that has a `.elizabeth-complete` marker and a reader in
`elizabeth.training.text`, and writes

    data/prepared/text/train.jsonl
    data/prepared/text/val.jsonl
    data/prepared/text/stats.json

one record per line in the OpenAI chat shape, split by a seeded hash of
the record (so a re-run, or a run on the box, draws the same line).
Exact duplicates are written once. A source whose licence forbids
training for release is refused, not included quietly; `stats.json`
says so, per source, next to its row count.

Run from the training venv: the parquet corpora need pyarrow, which the
runtime venv does not carry on purpose. Never commits anything —
data/prepared/ is gitignored, and this script writes that .gitignore if
it is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.training import text as prep
from elizabeth.training.manifest import Manifest

MANIFEST = REPO / "data" / "datasets.toml"
PREPARED_ROOT = REPO / "data" / "prepared"
# Everything under data/prepared/ is derived from corpora that are
# themselves gitignored, and is hundreds of MB. The ignore file is the
# one thing in that directory git may see.
GITIGNORE = "*\n!.gitignore\n"


def ensure_gitignored(out: Path) -> None:
    if PREPARED_ROOT in out.resolve().parents or out.resolve() == PREPARED_ROOT:
        marker = PREPARED_ROOT / ".gitignore"
        if not marker.exists():
            PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
            marker.write_text(GITIGNORE)


def _count(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _fmt(counter: dict[str, int]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(counter.items()))


def print_table(stats: dict) -> None:
    rows = [
        (
            name,
            str(s["n"]),
            str(s["train"]),
            str(s["val"]),
            str(s["duplicates"]),
            _fmt(s["lang"]),
            _fmt(s["kind"]),
            "yes" if s["publishable"] else "no",
            s["licence"][:28],
        )
        for name, s in stats["sources"].items()
    ]
    head = ("source", "n", "train", "val", "dup", "lang", "kind", "publish", "licence")
    widths = [max(len(r[i]) for r in (head, *rows)) for i in range(len(head))]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)))
    t = stats["totals"]
    print("-" * len(line))
    print(
        f"total n={t['n']} train={t['train']} val={t['val']}   "
        f"by lang: {_fmt(t['by_lang'])}   by kind: {_fmt(t['by_kind'])}"
    )
    print(f"publishable as one adapter: {'yes' if stats['publishable'] else 'NO'}")
    for name, why in stats["refused"].items():
        print(f"refused  {name}: {why}")
    for name in stats["missing"]:
        print(f"missing  {name}: no {prep.COMPLETE_MARKER} marker — run scripts/fetch_datasets.py")
    for name in stats["unreadable"]:
        print(f"no reader  {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", action="append", help="manifest name(s); repeatable")
    ap.add_argument("--limit", type=int, help="records per source, for a smoke run")
    ap.add_argument(
        "--publishable-only",
        action="store_true",
        help="drop every source whose weights_publishable is not 'yes'",
    )
    ap.add_argument("--seed", type=int, default=prep.DEFAULT_SEED)
    ap.add_argument("--val-fraction", type=float, default=prep.VAL_FRACTION)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--datasets-dir", type=Path, default=prep.DATASETS_DIR)
    ap.add_argument("--out", type=Path, default=prep.PREPARED_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    args = ap.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    try:
        plan = prep.plan(
            manifest,
            datasets_dir=args.datasets_dir,
            only=args.only,
            publishable_only=args.publishable_only,
        )
    except (prep.LicenceRefused, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    print(f"{len(plan.included)} sources to read: {', '.join(d.name for d in plan.included)}")
    for name, why in plan.refused.items():
        print(f"refused  {name}: {why}")
    for name in plan.missing:
        print(f"missing  {name}")
    if args.dry_run:
        return 0
    if not plan.included:
        print("nothing to prepare", file=sys.stderr)
        return 1

    ensure_gitignored(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "sources": {},
        "refused": plan.refused,
        "missing": plan.missing,
        "unreadable": plan.unreadable,
        "totals": {"n": 0, "train": 0, "val": 0, "by_lang": {}, "by_kind": {}},
        "publishable": all(prep.publishable(d) for d in plan.included),
    }
    seen: set[bytes] = set()
    with (
        (args.out / "train.jsonl").open("w", encoding="utf-8") as train,
        (args.out / "val.jsonl").open("w", encoding="utf-8") as val,
    ):
        for ds in plan.included:
            t0 = time.perf_counter()
            per: dict = {
                "n": 0,
                "train": 0,
                "val": 0,
                "duplicates": 0,
                "lang": {},
                "kind": {},
                "target": ds.target,
                "priority": ds.priority,
                "licence": ds.license,
                "flags": ds.flags,
                "publishable": prep.publishable(ds),
            }
            for rec in prep.load(ds, args.datasets_dir, limit=args.limit):
                canonical = rec.canonical()
                digest = hashlib.sha256(canonical.encode()).digest()
                if digest in seen:
                    per["duplicates"] += 1
                    continue
                seen.add(digest)
                side = prep.partition(canonical, args.seed, args.val_fraction)
                (train if side == "train" else val).write(
                    json.dumps(rec.as_dict(), ensure_ascii=False) + "\n"
                )
                per["n"] += 1
                per[side] += 1
                _count(per["lang"], rec.lang)
                _count(per["kind"], rec.kind)
                stats["totals"]["n"] += 1
                stats["totals"][side] += 1
                _count(stats["totals"]["by_lang"], rec.lang)
                _count(stats["totals"]["by_kind"], rec.kind)
            per["seconds"] = round(time.perf_counter() - t0, 1)
            # Rows the reader refused rather than guessed at. Written per
            # source so a corpus that quietly loses rows is visible in
            # stats.json instead of only in the arithmetic.
            per["dropped"] = prep.drops()
            stats["sources"][ds.name] = per
            print(f"{ds.name}: {per['n']} records in {per['seconds']}s", flush=True)
            for why, count in sorted(per["dropped"].items()):
                print(f"  dropped {count}: {why}", flush=True)

    (args.out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print()
    print_table(stats)
    print(f"\nwrote {args.out}/{{train,val}}.jsonl and stats.json")
    return 0 if stats["totals"]["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
