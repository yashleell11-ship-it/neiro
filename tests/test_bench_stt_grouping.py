"""EdACC is the only corpus we hold with a per-utterance accent label,
and a pooled WER over it would be a number about nobody — 23% Southern
British, 23% Mainstream US, the rest a long L2 tail. These pin the two
things that make it usable: the accent column survives the read, and the
scoring sentinels do not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("bench_stt", REPO / "scripts" / "bench_stt.py")
assert spec and spec.loader
bench_stt = importlib.util.module_from_spec(spec)
sys.modules["bench_stt"] = bench_stt
spec.loader.exec_module(bench_stt)

# pyarrow lives in the TRAINING venv, not the runtime one — which is
# precisely why bench_stt shells out to read shards instead of importing
# it. So these tests use that interpreter for both halves: writing the
# fixture and, as `reader_python`, reading it back.
TRAINING_PYTHON = REPO / "training" / ".venv" / "bin" / "python"

pytestmark = pytest.mark.skipif(
    not TRAINING_PYTHON.exists(), reason="needs the training venv, which has pyarrow"
)

_WRITER = """
import json, sys
import pyarrow as pa, pyarrow.parquet as pq
job = json.load(sys.stdin)
rows = [
    {k: ({"bytes": v["bytes"].encode(), "path": v["path"]} if isinstance(v, dict) else v)
     for k, v in row.items()}
    for row in job["rows"]
]
pq.write_table(pa.Table.from_pylist(rows), job["path"])
"""


def _shard(path: Path, rows: list[dict]) -> None:
    import json
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(TRAINING_PYTHON), "-c", _WRITER],
        input=json.dumps({"path": str(path), "rows": rows}).encode(),
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def _edacc_rows() -> list[dict]:
    # EdACC's real schema, confirmed against the live datasets-server:
    # speaker, text, accent, raw_accent, gender, l1, audio — where audio
    # is a {bytes, path} struct like every other corpus here.
    return [
        {"audio": {"bytes": "AAAA", "path": "a.wav"}, "text": "HELLO THERE",
         "accent": "Southern British English"},
        {"audio": {"bytes": "BBBB", "path": "b.wav"}, "text": "HOW ARE YOU",
         "accent": "Mainstream US English"},
        {"audio": {"bytes": "CCCC", "path": "c.wav"},
         "text": "IGNORE_TIME_SEGMENT_IN_SCORING", "accent": "Southern British English"},
    ]


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    root = tmp_path / "edacc"
    _shard(root / "data" / "validation-00000-of-00001.parquet", _edacc_rows())
    (root / ".neiro-complete").write_text("{}")
    return tmp_path


class TestEdaccGrouping:
    def test_the_accent_label_survives_the_read(self, corpus_root: Path) -> None:
        rows, skipped = bench_stt.load_rows(
            "edacc", limit=10, data_root=corpus_root, reader_python=TRAINING_PYTHON
        )
        assert not skipped, skipped
        accents = {g for _, _, g in rows}
        assert accents == {"Southern British English", "Mainstream US English"}

    def test_the_scoring_sentinel_is_dropped(self, corpus_root: Path) -> None:
        # Left in, IGNORE_TIME_SEGMENT_IN_SCORING is counted as four
        # reference words of pure deletion and wrecks the WER it exists
        # to protect.
        rows, _ = bench_stt.load_rows(
            "edacc", limit=10, data_root=corpus_root, reader_python=TRAINING_PYTHON
        )
        assert len(rows) == 2
        assert all("IGNORE_TIME_SEGMENT" not in t for _, t, _ in rows)

    def test_the_old_completion_marker_is_accepted(
        self, corpus_root: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # The box still runs the pre-rename fetcher, so its corpora carry
        # `.neiro-complete`. Warning "partial download" about 6.5 GB that
        # is demonstrably complete trains the reader to ignore warnings.
        bench_stt.load_rows(
            "edacc", limit=10, data_root=corpus_root, reader_python=TRAINING_PYTHON
        )
        assert "partial download" not in capsys.readouterr().err

    def test_a_corpus_without_a_group_column_still_reads(self, tmp_path: Path) -> None:
        # Every other corpus has no accent label; grouping must be
        # additive, not a new way for them to fail.
        root = tmp_path / "svarah-indic-accented-english"
        _shard(
            root / "data" / "test-00000-of-00001.parquet",
            [{"audio_filepath": {"bytes": "AAAA", "path": "a.wav"}, "text": "HELLO"}],
        )
        (root / ".elizabeth-complete").write_text("{}")
        rows, skipped = bench_stt.load_rows(
            "svarah", limit=10, data_root=tmp_path, reader_python=TRAINING_PYTHON
        )
        assert not skipped, skipped
        assert len(rows) == 1
        assert rows[0][2] is None, "no group column means no group, not a crash"
