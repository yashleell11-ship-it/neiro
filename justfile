default:
    @just --list

doctor:
    uv run neiro doctor

run:
    uv run neiro run

bench:
    uv run neiro bench

wer:
    uv run neiro wer

test:
    uv run pytest -q

lint:
    uv run ruff check .

# Gate G1 (Stage 0 Task 2): does ctranslate2 actually run int8 on this
# Blackwell GPU? Runs the spike twice — once with ~/.nv/ComputeCache
# cleared (forces any PTX JIT to happen and pays for it), once warm —
# so a multi-second first-load stall shows up as a number, not a mystery.
gate-g1:
    #!/usr/bin/env bash
    set -euo pipefail
    source env.sh
    echo "=== cold run (~/.nv/ComputeCache cleared — forces PTX JIT if any) ==="
    rm -rf ~/.nv/ComputeCache
    uv run python scripts/spike_blackwell.py | tee /tmp/neiro-g1-cold.log
    echo
    echo "=== warm run (cache populated by the run above) ==="
    uv run python scripts/spike_blackwell.py | tee /tmp/neiro-g1-warm.log
    echo
    cold=$(grep MODEL_LOAD_SECONDS /tmp/neiro-g1-cold.log | cut -d= -f2)
    warm=$(grep MODEL_LOAD_SECONDS /tmp/neiro-g1-warm.log | cut -d= -f2)
    echo "model load — cold: ${cold}s   warm: ${warm}s"
