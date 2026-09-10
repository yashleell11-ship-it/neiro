#!/usr/bin/env bash
# Source this — do not execute it — before any Python process that
# touches ctranslate2 or faster-whisper:
#
#     source env.sh
#     uv run python scripts/spike_blackwell.py
#
# CTranslate2's Linux wheel is built against CUDA 12.8 and dynamically
# loads libcublas.so.12 at runtime. On this box that means the copy
# pip-installed as nvidia-cublas-cu12, not a system CUDA toolkit (there
# isn't one). LD_LIBRARY_PATH must be set BEFORE the Python process
# starts and its CUDA extensions load — setting it from inside Python
# (os.environ[...] = ...) is too late. See Gate G1, docs/DECISIONS.md.
#
# Resolving the path itself needs no GPU access, just importing the
# already-installed nvidia-cublas-cu12 package to ask where its bundled
# .so lives — so this works even before LD_LIBRARY_PATH is set.
#
# nvidia.cublas is a PEP 420 namespace package (no __init__.py, because
# the nvidia-* wheels are split across several pip packages that share
# the `nvidia` namespace) — its `__file__` is always None. Use `__path__`
# instead; `os.path.dirname(l.__file__)` looks plausible and fails with
# a TypeError every time.

_neiro_cublas_dir="$(uv run python -c 'import nvidia.cublas.lib as l; print(l.__path__[0])' 2>/dev/null)"

if [ -n "${_neiro_cublas_dir}" ]; then
    export LD_LIBRARY_PATH="${_neiro_cublas_dir}:${LD_LIBRARY_PATH:-}"
else
    echo "env.sh: could not resolve nvidia-cublas-cu12's lib dir — is it installed? (uv sync)" >&2
fi

unset _neiro_cublas_dir
