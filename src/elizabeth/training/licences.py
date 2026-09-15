"""What the weights a training run produces may actually be used for.

A checkpoint inherits the most restrictive licence of anything it was
trained on, and nothing in a training loop notices. The manifest already
records the answer per corpus (`weights_publishable`); until this module
existed, no recipe read it — so a run that included RAVDESS
(CC-BY-NC-SA) or EARS (non-commercial) produced a `best.pt` that cannot
ship in an Apache-2.0 repo, with no error, no warning, and a report
indistinguishable from a clean one.

It lives here rather than in the recipe so it can be tested without
torch, and so any later recipe (STT fine-tune, the persona LoRA) answers
the question the same way.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from elizabeth.training.manifest import Manifest

MANIFEST = Path(__file__).resolve().parents[3] / "data" / "datasets.toml"

# Most restrictive wins, and an unanswered question is not a yes.
_ORDER = {"yes": 0, "unclear": 1, "no": 2}
_VERDICT = {v: k for k, v in _ORDER.items()}


def licences_for(corpora: Iterable[str], manifest_path: Path | None = None) -> dict:
    """What the weights this run produces may be used for.

    A checkpoint inherits the most restrictive licence of anything it was
    trained on, and nothing in the training loop notices: RAVDESS is
    CC-BY-NC-SA and EARS is non-commercial, so a run that quietly
    includes either produces a `best.pt` that cannot ship in an
    Apache-2.0 repo — with no error, no warning, and a report that looks
    identical to a clean one. The manifest already records this per
    corpus (`weights_publishable`); the recipe simply never read it.

    Returns the per-corpus table and one verdict for the run:
    "yes" only if every corpus says yes, "no" if any says no, else
    "unclear" — the same conservative ordering `elizabeth licences` uses.
    """
    manifest = Manifest.load(manifest_path or MANIFEST)
    known = {d.name: d for d in manifest.dataset}
    table, worst = {}, _ORDER["yes"]
    for name in corpora:
        entry = known.get(name)
        if entry is None:
            # A reader with no manifest entry is not evidence of
            # permission; it is an unanswered question.
            table[name] = {"licence": "unknown", "weights_publishable": "unclear"}
        else:
            table[name] = {
                "licence": entry.license,
                "weights_publishable": entry.weights_publishable,
            }
        worst = max(worst, _ORDER[table[name]["weights_publishable"]])
    return {"per_corpus": table, "weights_publishable": _VERDICT[worst]}
