"""Indexing emotion corpora into one table, and splitting it honestly.

The load-bearing test here is speaker-independence. CREMA-D's 91 actors
each record the same 12 sentences in 6 emotions; split those rows at
random and a model scores brilliantly by recognising the actor and the
scripted sentence rather than the emotion. SER results have been
inflated that way for years.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neiro.training.corpora import (
    READERS,
    Utterance,
    _bucket,
    load,
    read_crema_d,
    read_ravdess,
    read_savee,
    read_tess,
    split,
    summarise,
)


def _touch(root: Path, names: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / n).write_bytes(b"RIFF")
    return root


class TestCremaD:
    def test_parses_speaker_and_emotion_from_the_filename(self, tmp_path: Path) -> None:
        root = _touch(tmp_path / "crema-d", ["1001_DFA_ANG_XX.wav", "1078_IEO_SAD_LO.wav"])
        rows = read_crema_d(root)
        assert len(rows) == 2
        anger = next(r for r in rows if r.label == "anger")
        assert anger.speaker == "crema-d:1001"
        assert anger.raw_label == "ANG", "the corpus's own spelling is kept for debugging"
        assert anger.arousal > 0.5 and anger.valence < 0
        assert next(r for r in rows if r.label == "sad").arousal < 0

    def test_speaker_ids_are_namespaced_by_corpus(self, tmp_path: Path) -> None:
        # "1001" is a plausible id in more than one corpus; without the
        # namespace two different people would merge into one speaker and
        # break the split guarantee.
        root = _touch(tmp_path / "crema-d", ["1001_DFA_ANG_XX.wav"])
        assert read_crema_d(root)[0].speaker.startswith("crema-d:")

    def test_a_file_that_does_not_match_is_skipped_not_guessed(self, tmp_path: Path) -> None:
        root = _touch(tmp_path / "crema-d", ["README.wav", "1001_DFA_ANG_XX.wav"])
        assert len(read_crema_d(root)) == 1

    def test_an_empty_corpus_yields_nothing_rather_than_raising(self, tmp_path: Path) -> None:
        # A half-downloaded corpus must produce a visible count of zero,
        # not stop the whole indexing run.
        (tmp_path / "crema-d").mkdir()
        assert read_crema_d(tmp_path / "crema-d") == []


class TestOtherReaders:
    def test_ravdess_emotion_code_is_the_third_field(self, tmp_path: Path) -> None:
        root = _touch(tmp_path / "ravdess", ["03-01-05-01-02-01-12.wav"])
        rows = read_ravdess(root)
        assert len(rows) == 1
        assert rows[0].label == "anger" and rows[0].speaker == "ravdess:12"

    def test_tess_handles_the_two_word_emotion(self, tmp_path: Path) -> None:
        root = _touch(tmp_path / "tess", ["YAF_dog_pleasant_surprise.wav", "OAF_back_angry.wav"])
        labels = {r.arousal for r in read_tess(root)}
        assert len(read_tess(root)) == 2
        assert all(a > 0 for a in labels)

    def test_savee_multi_letter_codes(self, tmp_path: Path) -> None:
        # "sa" (sad) and "su" (surprise) both start with s — a naive
        # single-letter parse would call every sad file a surprise.
        root = _touch(tmp_path / "savee", ["DC_sa01.wav", "DC_su01.wav", "DC_a01.wav"])
        rows = read_savee(root)
        assert {r.label for r in rows} == {"sad", "surprise", "anger"}
        assert next(r for r in rows if r.label == "sad").arousal < 0
        assert next(r for r in rows if r.label == "surprise").arousal > 0

    def test_every_reader_is_registered(self) -> None:
        assert set(READERS) == {"crema-d", "ravdess", "tess", "savee"}


class TestSplit:
    def _rows(self, n_speakers: int = 40) -> list[Utterance]:
        return [
            Utterance(f"/x/{s}_{i}.wav", -0.6, 0.8, "crema-d", f"crema-d:{1000 + s}", "anger")
            for s in range(n_speakers)
            for i in range(6)
        ]

    def test_no_speaker_appears_in_two_partitions(self) -> None:
        train, val, test = split(self._rows())
        a, b, c = ({r.speaker for r in part} for part in (train, val, test))
        assert not (a & b) and not (a & c) and not (b & c)

    def test_every_row_lands_somewhere(self) -> None:
        rows = self._rows()
        train, val, test = split(rows)
        assert len(train) + len(val) + len(test) == len(rows)

    def test_the_split_is_stable_across_runs(self) -> None:
        # Python's hash() is salted per process; using it would reshuffle
        # the split every run and leak test speakers into train over time.
        rows = self._rows()
        assert [len(p) for p in split(rows)] == [len(p) for p in split(rows)]
        assert _bucket("crema-d:1001") == _bucket("crema-d:1001")

    def test_different_speakers_land_in_different_buckets(self) -> None:
        buckets = {_bucket(f"crema-d:{1000 + i}") for i in range(50)}
        assert len(buckets) > 20, "the hash should spread speakers, not clump them"

    def test_an_empty_input_splits_cleanly(self) -> None:
        assert split([]) == ([], [], [])


class TestSummary:
    def test_counts_make_a_silent_problem_visible(self) -> None:
        rows = [
            Utterance("/a.wav", -0.6, 0.8, "crema-d", "crema-d:1", "anger"),
            Utterance("/b.wav", 0.8, 0.5, "crema-d", "crema-d:2", "happy"),
        ]
        s = summarise(rows)
        assert s["n"] == 2 and s["speakers"] == 2
        assert s["by_corpus"] == {"crema-d": 2}
        assert s["acted"] == 2 and s["natural"] == 0

    def test_acted_and_natural_are_counted_apart(self) -> None:
        # One accuracy over a mix of the two, unlabelled, is the
        # commonest way SER numbers mislead.
        rows = [
            Utterance("/a.wav", -0.6, 0.8, "crema-d", "s1", "anger"),
            Utterance("/b.wav", -0.6, 0.8, "msp-podcast", "s2", "anger"),
            Utterance("/c.wav", -0.6, 0.8, "mystery-corpus", "s3", "anger"),
        ]
        s = summarise(rows)
        assert (s["acted"], s["natural"], s["unrecorded"]) == (1, 1, 1)

    def test_empty(self) -> None:
        assert summarise([])["n"] == 0


class TestRealDownload:
    """Runs against whatever is actually in data/datasets/."""

    def test_whatever_is_downloaded_indexes_without_error(self) -> None:
        rows = load()
        s = summarise(rows)
        assert s["n"] >= 0
        if rows:
            # Every indexed row must carry a usable label, by construction.
            assert all(-1 <= r.valence <= 1 and -1 <= r.arousal <= 1 for r in rows)
            assert all(r.speaker for r in rows)

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data/datasets/crema-d").is_dir(),
        reason="CREMA-D not downloaded yet",
    )
    def test_labels_are_canonical_across_corpora(self) -> None:
        # Mixing corpora is the point of this module; it only works if
        # every corpus's spelling has already collapsed to one name.
        rows = load()
        if rows:
            assert all(r.label.islower() for r in rows), (
                "labels must be canonical, not corpus spellings"
            )

    def test_crema_d_has_its_published_shape(self) -> None:
        # 91 actors, 6 emotions, ~7442 clips. If the count collapses, the
        # download or the parser broke — and a training run on 12 rows
        # would otherwise just look like a bad model.
        s = summarise(load(["crema-d"]))
        assert s["speakers"] == 91
        assert s["n"] > 7000
        assert len(s["by_label"]) == 6
