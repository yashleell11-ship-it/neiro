#!/usr/bin/env python3
"""Can the current token actually PULL from each gated dataset?

    uv run scripts/check_gated_access.py

Written because two easier checks both lied on 2026-09-13:

  - `dataset_info(repo, token=...)` succeeds on a gated repo. It reads
    public metadata and says nothing about file access.
  - `hf_hub_download(repo, "README.md")` also succeeds — every
    AI4Bharat repo has a public README behind a gated data directory.

Both reported eight datasets as reachable. Every one of them 403'd on
the first real file. So this downloads an actual DATA file: the smallest
parquet/tar/zip in the repo, which is the only thing that proves
anything.

It also names the likeliest cause when everything fails at once, because
that cause is invisible and costs an hour otherwise: `hf auth login`
defaults to a browser OAuth flow, and an **oauth session token carries
no gated-repo scope**. A personal READ token from
huggingface.co/settings/tokens is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neiro.training.manifest import Manifest

DATA_SUFFIXES = (".parquet", ".tar", ".tar.gz", ".tgz", ".zip", ".arrow", ".wav", ".flac")


def main() -> int:
    from huggingface_hub import get_token, hf_hub_download, list_repo_tree, whoami
    from huggingface_hub.errors import GatedRepoError

    token = get_token()
    if not token:
        print("No token. Run: uv run hf auth login --token <a READ token>")
        return 2

    kind = "unknown"
    try:
        kind = (whoami(token).get("auth") or {}).get("accessToken", {}).get("type", "unknown")
    except Exception as exc:  # noqa: BLE001 — an unreadable type is not fatal
        kind = f"unreadable ({type(exc).__name__})"
    print(f"token type: {kind}")
    if kind == "oauth" or token.startswith("hf_oauth"):
        print(
            "  ^ this is a browser SESSION token. It reads public metadata fine and "
            "cannot pull a single gated file.\n"
            "    Make a READ token at https://huggingface.co/settings/tokens, then:\n"
            "      uv run hf auth login --token <that token>"
        )

    manifest = Manifest.load(REPO / "data" / "datasets.toml")
    gated = [
        d for d in manifest.dataset if d.is_hf and d.access in ("hf-login", "hf-gated-approval")
    ]
    print(f"\nchecking {len(gated)} gated datasets with a REAL data file:\n")

    blocked = []
    for d in gated:
        try:
            entries = list_repo_tree(d.hf_id, repo_type="dataset", recursive=True, token=token)
            smallest = min(
                (e for e in entries if str(getattr(e, "path", "")).endswith(DATA_SUFFIXES)),
                key=lambda e: (
                    getattr(e, "size", None)
                    or getattr(getattr(e, "lfs", None), "size", 0)
                    or 1 << 62
                ),
                default=None,
            )
            if smallest is None:
                print(f"  ?      {d.name:<28} no data file in the tree")
                continue
            hf_hub_download(d.hf_id, smallest.path, repo_type="dataset", token=token)
            print(f"  OK     {d.name:<28} {d.hf_id}")
        except GatedRepoError:
            blocked.append(d)
            print(f"  GATED  {d.name:<28} {d.hf_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {type(exc).__name__:<6} {d.name:<28} {d.hf_id}")

    if blocked:
        print(f"\n{len(blocked)} still blocked. Accept at:")
        for d in blocked:
            print(f"  https://huggingface.co/datasets/{d.hf_id}")
        if kind == "oauth" or token.startswith("hf_oauth"):
            print("\n...but with an oauth token, clicking will not help. Fix the token first.")
    else:
        print("\nAll gated datasets are pullable. `neiro fetch-datasets --tier 1` will work.")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
