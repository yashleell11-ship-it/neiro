"""Evaluations that score a swappable component against a committed
fixture set.

Deliberate deviation from the plan's sketched layout, which put these in
a root-level `evals/` directory: they live inside the package instead so
they are importable (`elizabeth eval --all` is a planned CLI command) and
unit-testable. The datasets they score against still live outside the
package, under `data/`.

Rule (CLAUDE.md): no model in modelspec.py changes without a
before/after table from these, recorded in docs/DECISIONS.md.
"""
