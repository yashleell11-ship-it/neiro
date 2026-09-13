"""Tests for scripts/fetch_datasets.py — the "did anything real arrive?" gate.

`_has_real_data` exists so that a snapshot holding only a loader script
and a README is reported as `no-data` instead of being marked complete.
It was inoperative for every HF dataset: `snapshot_download` always
fetches `.gitattributes`, and `Path(".gitattributes").suffix` is `""`,
so the ".gitattributes" entry in the metadata-suffix set matched nothing
and the file counted as data. Worst case was not `too-small` but `done`
plus a written marker, because entries recorded at `size_gb = 0.0` also
skip the size floor — a silently empty corpus at training time.

The downloader is never run here; the on-disk shape of a snapshot is all
the gate looks at, so tmp_path is the whole fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from neiro.training.manifest import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_datasets as fd


def _dataset(**overrides: object) -> Dataset:
    base: dict[str, object] = {
        "name": "x",
        "target": "hindi_emotion_text",
        "priority": 1,
        "hf_id": "org/x",
        "license": "MIT",
        "flags": ["permissive"],
        "access": "direct",
        "size_gb": 1.0,
        "weights_publishable": "yes",
    }
    return Dataset(**(base | overrides))  # type: ignore[arg-type]


class TestMetadataIsNotData:
    def test_gitattributes_and_readme_are_not_data(self, tmp_path: Path) -> None:
        # The minimum every HF snapshot contains, data or not.
        (tmp_path / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs\n")
        (tmp_path / "README.md").write_text("# x")
        assert not fd._has_real_data(tmp_path)

    def test_a_loader_script_repo_as_snapshot_download_leaves_it(self, tmp_path: Path) -> None:
        # Exactly the shape the docstring names: vctk.py + README, plus the
        # .gitattributes the hub adds. The data lives behind the script.
        (tmp_path / ".gitattributes").write_text("* text=auto\n")
        (tmp_path / "vctk.py").write_text("class Vctk: pass")
        (tmp_path / "README.md").write_text("# VCTK")
        (tmp_path / "dataset_infos.json").write_text("{}")
        assert not fd._has_real_data(tmp_path)

    def test_git_housekeeping_dotfiles_are_not_data(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.lock\n")
        (tmp_path / "README.md").write_text("# x")
        assert not fd._has_real_data(tmp_path)

    def test_the_datasets_library_sidecars_are_not_data(self, tmp_path: Path) -> None:
        # `save_to_disk()` writes these next to the arrow shards. Without
        # the shards they describe nothing.
        for name in ("dataset_dict.json", "dataset_info.json", "state.json"):
            (tmp_path / name).write_text("{}")
        (tmp_path / ".gitattributes").write_text("")
        (tmp_path / "README.md").write_text("# x")
        assert not fd._has_real_data(tmp_path)


class TestPayloadsAreData:
    def test_a_json_payload_is_data(self, tmp_path: Path) -> None:
        # glaive-function-calling-v2 is one .json file and nothing else.
        # Treating every .json as metadata would report it as no-data.
        (tmp_path / ".gitattributes").write_text("")
        (tmp_path / "README.md").write_text("# x")
        (tmp_path / "glaive-function-calling-v2.json").write_bytes(b"[]")
        assert fd._has_real_data(tmp_path)

    @pytest.mark.parametrize(
        "name", ["train.parquet", "audio.tar", "clip.wav", "PHINC.csv", "dialogs.jsonl"]
    )
    def test_real_payloads_count_beside_the_hub_metadata(self, tmp_path: Path, name: str) -> None:
        (tmp_path / ".gitattributes").write_text("")
        (tmp_path / "README.md").write_text("# x")
        (tmp_path / name).write_bytes(b"\x00" * 16)
        assert fd._has_real_data(tmp_path)


class TestFetchRefusesAnEmptySnapshot:
    def test_a_zero_size_entry_with_only_metadata_is_no_data_not_done(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The worst case: `size_gb = 0.0` skips the size floor, so the
        # only thing between an empty snapshot and a completion marker is
        # `_has_real_data`. Before the fix this returned "done" and wrote
        # the marker.
        def fake_fetch_hf(ds: Dataset, dest: Path, token: str | None) -> str:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".gitattributes").write_text("* text=auto\n")
            (dest / "README.md").write_text("# x")
            return "done"

        monkeypatch.setattr(fd, "fetch_hf", fake_fetch_hf)
        dest = tmp_path / "x"
        assert fd.fetch(_dataset(size_gb=0.0), dest, None, force=False) == "no-data"
        assert not (dest / fd.MARKER).exists(), "an empty snapshot must never be marked complete"
        assert "no-data" in fd.REMEDY
