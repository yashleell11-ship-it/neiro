# Elizabeth's commands. `just` is not installed yet (needs sudo), so every
# recipe here is a plain command you can also paste directly — that is
# deliberate: a task runner should save typing, not become a dependency
# for running the project at all.

default:
    @just --list

# --- the ones you run most -------------------------------------------

# Check every assumption, naming the fix for each failure.
doctor:
    uv run elizabeth doctor

# The whole emotional loop for one recording: prosody, z-scores, the
# annotation the prompt would get, her face weights, her voice.
affect WAV:
    uv run elizabeth affect {{WAV}}

# Hear her say something. An <e:LABEL:D> tag is honoured.
say TEXT:
    uv run elizabeth say "{{TEXT}}"

# The whole loop: SPACE, speak, SPACE, hear her. SPACE while she speaks interrupts.
talk:
    #!/usr/bin/env bash
    source env.sh && uv run elizabeth talk

# --- data -------------------------------------------------------------

fetch-models:
    uv run elizabeth fetch-models --runs local --purpose runtime

fetch-datasets TIER="1":
    uv run elizabeth fetch-datasets --tier {{TIER}}

# Can this token actually pull the gated corpora? Downloads a real data
# file, because dataset_info() and README.md both succeed on a gated repo.
check-access:
    uv run scripts/check_gated_access.py

# --- gates ------------------------------------------------------------

# G2: does everything fit in 7730 MiB at once?
gate-vram:
    #!/usr/bin/env bash
    source env.sh && uv run scripts/spike_vram.py

# G3b rehearsal on acted corpora. The real gate is Yash's own voice.
gate-arousal:
    uv run scripts/spike_arousal.py

# Stage 1 gate: >=98% of replies open with a well-formed emotion tag.
gate-tags N="100":
    uv run scripts/eval_tag_compliance.py --n {{N}}

# Does the [voice: ...] annotation actually change what she says?
persona:
    uv run scripts/eval_persona.py

# G3a: WER on Yash's own recordings. Needs `elizabeth record-set` first.
wer:
    #!/usr/bin/env bash
    source env.sh && uv run elizabeth wer

# Replay fixture turns through fakes: measures the PIPELINE, not the GPU.
bench:
    uv run scripts/bench_turn.py

# The same, failing if the pipeline itself has gained overhead.
bench-assert:
    uv run scripts/bench_turn.py --assert

# WER against a public corpus — an honest prior before recording his own.
bench-stt CORPUS="svarah":
    #!/usr/bin/env bash
    source env.sh && uv run scripts/bench_stt.py --corpus {{CORPUS}}

# --- training (separate venv, CUDA torch) ------------------------------

train-ser CORPORA="crema-d ravdess rasa":
    cd training && uv run python recipes/ser_train.py --corpora {{CORPORA}}

extract-rasa:
    cd training && uv run python ../scripts/extract_rasa.py

# --- development -------------------------------------------------------

test:
    uv run pytest -q

lint:
    uv run ruff check src tests scripts training/recipes
    uv run ruff format --check src tests scripts training/recipes

fix:
    uv run ruff check --fix src tests scripts training/recipes
    uv run ruff format src tests scripts training/recipes

# Everything that must be green before a commit.
check: lint test bench-assert doctor
