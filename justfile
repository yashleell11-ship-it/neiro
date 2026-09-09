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
