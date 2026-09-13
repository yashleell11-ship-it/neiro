"""Indexing emotion corpora into one table, and splitting it honestly.

The load-bearing test here is speaker-independence. CREMA-D's 91 actors
each record the same 12 sentences in 6 emotions; split those rows at
random and a model scores brilliantly by recognising the actor and the
scripted sentence rather than the emotion. SER results have been
inflated that way for years.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neiro.training.corpora import (
    DATASETS_DIR,
    EMONET_INDEX,
    EMONET_INDEX_COLUMNS,
    READERS,
    Utterance,
    _bucket,
    emonet_agreed,
    load,
    parse_emonet_label,
    read_crema_d,
    read_emonet_voice_bench,
    read_rasa,
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
        assert set(READERS) == {
            "crema-d",
            "ravdess",
            "tess",
            "savee",
            "rasa",
            "emonet-voice-bench",
        }


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

    def test_synthetic_is_a_third_bucket_not_an_unrecorded_one(self) -> None:
        # TTS-generated speech is counted on its own: folding it into
        # "unrecorded" would hide that a corpus is synthetic, folding it
        # into acted would let it inflate the acted number.
        rows = [
            Utterance("/a.wav", -0.6, 0.8, "crema-d", "s1", "anger"),
            Utterance("/b.wav", -0.6, 0.8, "emonet-voice-bench", "s2", "anger"),
            Utterance("/c.wav", -0.6, 0.8, "emonet-voice-bench", "s3", "anger"),
        ]
        s = summarise(rows)
        assert (s["acted"], s["natural"], s["synthetic"], s["unrecorded"]) == (1, 0, 2, 0)
        assert rows[1].kind == "synthetic" and rows[1].acted is None

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

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[1] / "data/datasets/crema-d").is_dir(),
        reason="CREMA-D not downloaded yet",
    )
    def test_crema_d_has_its_published_shape(self) -> None:
        # 91 actors, 6 emotions, ~7442 clips. If the count collapses, the
        # download or the parser broke — and a training run on 12 rows
        # would otherwise just look like a bad model.
        s = summarise(load(["crema-d"]))
        assert s["speakers"] == 91
        assert s["n"] > 7000
        assert len(s["by_label"]) == 6


class TestRasa:
    """AI4Bharat Rasa — the only Indian, emotional, CC-BY corpus here."""

    def test_it_parses_the_extracted_filenames(self, tmp_path: Path) -> None:
        root = _touch(tmp_path / "rasa", ["female_ANGER_00001.wav", "male_SAD_00002.wav"])
        rows = read_rasa(root)
        assert len(rows) == 2
        anger = next(r for r in rows if r.label == "anger")
        assert anger.speaker == "rasa:female"
        assert anger.arousal > 0.5

    def test_gender_stands_in_for_speaker(self, tmp_path: Path) -> None:
        # Rasa ships one male and one female voice per language and no
        # speaker ids, so gender is the only speaker axis there is — and
        # it is what the speaker-independent split actually needs.
        root = _touch(tmp_path / "rasa", ["male_HAPPY_00001.wav", "female_HAPPY_00002.wav"])
        assert len({r.speaker for r in read_rasa(root)}) == 2

    def test_reading_styles_are_not_in_the_extracted_set(self, tmp_path: Path) -> None:
        # WIKI, BOOK, NEWS and the rest are reading registers, not
        # emotions. A Wikipedia read is a different task, not a neutral
        # emotional state, so the extractor skips them entirely.
        root = _touch(tmp_path / "rasa", ["female_WIKI_00001.wav", "male_CONV_00002.wav"])
        assert read_rasa(root) == []

    def test_it_is_registered(self) -> None:
        assert "rasa" in READERS


def _emonet(root: Path, rows: list[tuple[str, str, str, str]], touch: bool = True) -> Path:
    """A corpus root shaped the way scripts/extract_emonet.py leaves it:
    `extracted/index.csv` plus (unless `touch` is off) the wavs it names.
    """
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with (extracted / EMONET_INDEX).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(EMONET_INDEX_COLUMNS)
        writer.writerows(rows)
    if touch:
        for name, *_ in rows:
            (extracted / name).write_bytes(b"RIFF")
    return root


class TestEmoNetLabelCell:
    """The parquet's `label` column: a repr of per-rater dicts, category
    URL-encoded, one 0/1/2 score per expert."""

    def test_it_decodes_the_category_and_keeps_every_score(self) -> None:
        raw = (
            "[{'human-1': {'Impatience%20and%20Irritability': 2}}, "
            "{'human-4': {'Impatience%20and%20Irritability': 0}}]"
        )
        assert parse_emonet_label(raw) == ("Impatience and Irritability", (2, 0))
        assert parse_emonet_label("[{'human-2': {'Jealousy%20%26%20Envy': 1}}]") == (
            "Jealousy & Envy",
            (1,),
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not a list",
            "[]",
            "[{'human-1': {'Shame': 1}}, {'human-2': {'Fear': 2}}]",  # two categories
            "[{'human-1': 'Shame'}]",
            "[{'human-1': {'Shame': '2'}}]",
            "[{'human-1': {'Shame': True}}]",
            "__import__('os')",
        ],
    )
    def test_anything_ambiguous_is_none_not_a_guess(self, raw: str) -> None:
        assert parse_emonet_label(raw) is None

    def test_agreement_means_every_rater_heard_it(self) -> None:
        # (2, 0) is a synthesis one expert heard and one did not. The
        # published bench keeps only unanimous rows, and so does this.
        assert emonet_agreed((1, 2, 2))
        assert emonet_agreed((1,))
        assert not emonet_agreed((2, 0))
        assert not emonet_agreed((2, 2, 0))
        assert not emonet_agreed((0, 0, 0))
        assert not emonet_agreed(())


# Two index rows shaped exactly as the extractor writes them.
_EMONET_ROWS = [
    ("dd53077f_enhanced-1a2b3c4d.wav", "dd53077f", "Shame", "1;2;2"),
    ("50e55b50_enhanced_4-5e6f7a8b.wav", "50e55b50", "Impatience and Irritability", "2;2"),
]


class TestEmoNet:
    """EmoNet-Voice Bench — synthetic English in 42 fine categories."""

    def test_it_reads_the_index_the_extractor_writes(self, tmp_path: Path) -> None:
        rows = read_emonet_voice_bench(_emonet(tmp_path / "emonet-voice-bench", _EMONET_ROWS))
        assert len(rows) == 2
        shame = next(r for r in rows if r.raw_label == "Shame")
        assert shame.label == "ashamed" and shame.valence < 0
        assert shame.speaker == "emonet-voice-bench:dd53077f"
        assert shame.path.endswith("extracted/dd53077f_enhanced-1a2b3c4d.wav")
        irritated = next(r for r in rows if r.label == "frustrated")
        assert irritated.raw_label == "Impatience and Irritability", (
            "the corpus's own spelling, decoded, is kept for debugging"
        )
        assert irritated.arousal > 0 and irritated.valence < 0

    def test_a_clip_not_every_rater_heard_is_dropped(self, tmp_path: Path) -> None:
        rows = [
            ("a-00000001.wav", "a", "Anger", "2;0"),
            ("b-00000002.wav", "b", "Anger", "2;2;0"),
            ("c-00000003.wav", "c", "Anger", "0;0;0"),
            ("d-00000004.wav", "d", "Anger", "1;2"),
            ("e-00000005.wav", "e", "Anger", "garbage"),
        ]
        out = read_emonet_voice_bench(_emonet(tmp_path / "emonet-voice-bench", rows))
        assert [r.speaker for r in out] == ["emonet-voice-bench:d"]

    def test_a_category_with_no_point_is_dropped_not_called_neutral(self, tmp_path: Path) -> None:
        rows = [
            ("a-00000001.wav", "a", "Authenticity", "2;2;2"),
            ("b-00000002.wav", "b", "Contentment", "2;2;2"),
        ]
        out = read_emonet_voice_bench(_emonet(tmp_path / "emonet-voice-bench", rows))
        assert [r.label for r in out] == ["content"]

    def test_a_row_whose_wav_is_missing_is_skipped(self, tmp_path: Path) -> None:
        # A half-written extraction must yield fewer rows, not a path
        # that fails in the middle of an epoch.
        root = _emonet(tmp_path / "emonet-voice-bench", _EMONET_ROWS, touch=False)
        assert read_emonet_voice_bench(root) == []

    def test_no_index_means_no_rows(self, tmp_path: Path) -> None:
        (tmp_path / "emonet-voice-bench" / "data").mkdir(parents=True)
        assert read_emonet_voice_bench(tmp_path / "emonet-voice-bench") == []

    def test_the_filename_prefix_is_the_pseudo_speaker(self, tmp_path: Path) -> None:
        # No voice id ships with the corpus; the 8-hex prefix shared by
        # the segments of one generation is the only identity there is.
        # Two segments of one generation must land on ONE side of the
        # split, so they must share a speaker.
        rows = [
            ("47ea8bf3_enhanced_1-aaaaaaaa.wav", "47ea8bf3", "Fear", "2;2"),
            ("47ea8bf3_enhanced_2-bbbbbbbb.wav", "47ea8bf3", "Fear", "1;2"),
            ("c9b81c42_enhanced-cccccccc.wav", "c9b81c42", "Fear", "2;2"),
        ]
        out = read_emonet_voice_bench(_emonet(tmp_path / "emonet-voice-bench", rows))
        assert len({r.speaker for r in out}) == 2
        assert all(r.speaker.startswith("emonet-voice-bench:") for r in out)

    def test_it_is_synthetic_not_acted(self, tmp_path: Path) -> None:
        out = read_emonet_voice_bench(_emonet(tmp_path / "emonet-voice-bench", _EMONET_ROWS))
        assert all(r.kind == "synthetic" and r.acted is None for r in out)
        s = summarise(out)
        assert s["synthetic"] == 2 and s["acted"] == 0 and s["unrecorded"] == 0

    def test_load_finds_it_by_directory_name(self, tmp_path: Path) -> None:
        _emonet(tmp_path / "emonet-voice-bench", _EMONET_ROWS)
        assert len(load(["emonet-voice-bench"], datasets_dir=tmp_path)) == 2

    @pytest.mark.skipif(
        not (DATASETS_DIR / "emonet-voice-bench" / "extracted" / EMONET_INDEX).is_file(),
        reason="EmoNet-Voice Bench not extracted yet (scripts/extract_emonet.py)",
    )
    def test_the_extracted_corpus_has_its_published_shape(self) -> None:
        # 12,600 rows rated, 4,692 unanimous — the bench size the paper
        # reports — minus the seven categories with no circumplex point.
        # If this collapses, the extraction or the index broke, and a
        # training run on a few hundred rows would just look like a bad
        # model.
        s = summarise(load(["emonet-voice-bench"]))
        assert 3500 < s["n"] < 4692
        assert s["synthetic"] == s["n"]
        assert s["speakers"] > 3000, "one pseudo-speaker per generation, not per corpus"
        assert "neutral" not in s["by_label"], "nothing here is ever called neutral"
