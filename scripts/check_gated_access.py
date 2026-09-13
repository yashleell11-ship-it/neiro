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


def _classify(token: str, whoami) -> tuple[str, str]:
    """Name the token kind, and say why it cannot read a gated repo.

    Three kinds have now each cost an hour here, and none of them
    announce themselves — every one reads public metadata perfectly and
    403s on the first real file:

      1. an **oauth session token** (`hf_oauth…`), which is what
         `hf auth login` makes by default via the browser flow;
      2. a **fine-grained token scoped to your own account only** —
         `canReadGatedRepos: true` looks right, but the repo permissions
         are scoped to one entity, so someone else's gated repo is still
         out of reach;
      3. no token at all.

    A classic **Read** token is the one that works.
    """
    fix = (
        "    Make a classic READ token (NOT fine-grained) at\n"
        "    https://huggingface.co/settings/tokens, then:\n"
        "      uv run hf auth login --token <that token>"
    )
    if token.startswith("hf_oauth"):
        return (
            "oauth session token",
            "  ^ a browser SESSION token. It reads public metadata and cannot pull "
            "one gated file.\n" + fix,
        )
    try:
        auth = (whoami(token).get("auth") or {}) or {}
        access = auth.get("accessToken") or {}
        role = access.get("role", "?")
        if role != "fineGrained":
            return (f"{role} token", "")
        fine = access.get("fineGrained") or {}
        scoped = fine.get("scoped") or []
        entities = [str((e.get("entity") or {}).get("name", "?")) for e in scoped]
        if fine.get("canReadGatedRepos") and entities:
            return (
                f"fine-grained, scoped to {', '.join(entities)}",
                "  ^ canReadGatedRepos is true, but the repo permissions are scoped to "
                f"{', '.join(entities)} only.\n"
                "    A gated dataset owned by someone else (ai4bharat, ARTPARK-IISc) is "
                "still out of reach.\n" + fix,
            )
        return ("fine-grained", "  ^ fine-grained tokens rarely cover others' gated repos.\n" + fix)
    except Exception as exc:  # noqa: BLE001 — an unreadable token is not fatal
        return (f"unreadable ({type(exc).__name__})", "")


def main() -> int:
    from huggingface_hub import get_token, hf_hub_download, list_repo_tree, whoami
    from huggingface_hub.errors import GatedRepoError

    token = get_token()
    if not token:
        print("No token. Run: uv run hf auth login --token <a READ token>")
        return 2

    kind, diagnosis = _classify(token, whoami)
    print(f"token: {kind}")
    if diagnosis:
        print(diagnosis)

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
        print(f"\n{len(blocked)} still blocked. On each page, scroll to the grey box")
        print('  ("You need to agree to share your contact information...") and press')
        print('  "Agree and access repository". Opening the page is not enough.\n')
        for d in blocked:
            print(f"  https://huggingface.co/datasets/{d.hf_id}")
        print(
            "\n  Verify afterwards at https://huggingface.co/settings/gated-repos —\n"
            "  it lists every gate this account has accepted. If a dataset is not\n"
            "  there, the click did not register (usually: signed in as someone else,\n"
            "  or the button was never pressed)."
        )
        if kind == "oauth" or token.startswith("hf_oauth"):
            print("\n...but with an oauth token, clicking will not help. Fix the token first.")
    else:
        print("\nAll gated datasets are pullable. `neiro fetch-datasets --tier 1` will work.")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
