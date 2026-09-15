"""What a checkpoint may be used for, decided before it is written.

Weights inherit the most restrictive licence of everything they were
trained on. Nothing in a training loop notices that, and a report from a
run that quietly included a non-commercial corpus looks exactly like a
report from a clean one — which is how a model that cannot ship ends up
shipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elizabeth.training.licences import MANIFEST, licences_for

CLEAN = """
schema_version = 1

[[dataset]]
name = "open-one"
target = "ser_lane_b"
priority = 1
hf_id = "someone/open-one"
license = "cc-by-4.0"
flags = []
access = "direct"
size_gb = 1.0
weights_publishable = "yes"

[[dataset]]
name = "noncommercial"
target = "ser_lane_b"
priority = 2
hf_id = "someone/noncommercial"
license = "cc-by-nc-sa-4.0"
flags = ["NC"]
access = "direct"
size_gb = 1.0
weights_publishable = "no"

[[dataset]]
name = "murky"
target = "ser_lane_b"
priority = 3
hf_id = "someone/murky"
license = "unstated"
flags = []
access = "direct"
size_gb = 1.0
weights_publishable = "unclear"
"""


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "datasets.toml"
    path.write_text(CLEAN)
    return path


class TestTheVerdict:
    def test_all_open_is_publishable(self, manifest: Path) -> None:
        assert licences_for(["open-one"], manifest)["weights_publishable"] == "yes"

    def test_one_non_commercial_corpus_decides_the_whole_run(self, manifest: Path) -> None:
        # The point of the module: 99 open corpora and one NC corpus is
        # an NC checkpoint, and nothing else about the run says so.
        got = licences_for(["open-one", "noncommercial"], manifest)
        assert got["weights_publishable"] == "no"

    def test_unclear_is_not_a_yes(self, manifest: Path) -> None:
        assert licences_for(["open-one", "murky"], manifest)["weights_publishable"] == "unclear"

    def test_no_beats_unclear_whatever_the_order(self, manifest: Path) -> None:
        forwards = licences_for(["murky", "noncommercial"], manifest)
        backwards = licences_for(["noncommercial", "murky"], manifest)
        assert forwards["weights_publishable"] == "no"
        assert backwards["weights_publishable"] == "no"

    def test_a_corpus_the_manifest_never_heard_of_is_unclear(self, manifest: Path) -> None:
        # A reader with no manifest entry is an unanswered question, not
        # permission. Corpora arrive as readers before they arrive as
        # manifest rows, and that gap is exactly when this would be wrong.
        got = licences_for(["open-one", "invented-corpus"], manifest)
        assert got["per_corpus"]["invented-corpus"]["weights_publishable"] == "unclear"
        assert got["weights_publishable"] == "unclear"

    def test_nothing_trained_on_is_vacuously_publishable(self, manifest: Path) -> None:
        assert licences_for([], manifest)["weights_publishable"] == "yes"


class TestTheTable:
    def test_every_corpus_is_named_with_its_licence(self, manifest: Path) -> None:
        table = licences_for(["open-one", "noncommercial"], manifest)["per_corpus"]
        assert set(table) == {"open-one", "noncommercial"}
        assert table["noncommercial"]["licence"] == "cc-by-nc-sa-4.0"


class TestAgainstTheRealManifest:
    def test_the_corpora_lane_b_trains_on_today(self) -> None:
        # RAVDESS is CC-BY-NC-SA, so every checkpoint trained with it is
        # private. This is the run that actually happened on 2026-09-14.
        got = licences_for(["crema-d", "ravdess", "rasa"], MANIFEST)
        assert got["per_corpus"]["ravdess"]["weights_publishable"] == "no"
        assert got["weights_publishable"] == "no"
        # ...and without it, the same three-corpus recipe is shippable.
        assert licences_for(["crema-d", "rasa"], MANIFEST)["weights_publishable"] == "yes"
