"""The rules the Hindi STT fine-tune's data path is built on.

`training/recipes/stt_train.py` needs CUDA torch and cannot be imported
in this venv, which is exactly why everything worth asserting lives in
`elizabeth.training.stt_data` instead. The load-bearing test here is the
speaker split: Kathbath publishes a `valid` shard that looks like a
held-out set and shares every one of its 20 speakers with `train`, so a
fine-tune that trusted the filename would report a WER for speakers it
had just trained on and there would be nothing in the output to say so.

The parquet tests build tiny synthetic shards in `tmp_path` — never the
real corpora, which are 63 GB — and skip where pyarrow is absent
(it lives in the training venv, the same reason `scripts/bench_stt.py`
shells out for its reads). Run them with the training interpreter:

    training/.venv/bin/python -m pytest tests/test_stt_train.py
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest

from elizabeth.training.corpora import _bucket
from elizabeth.training.stt_data import (
    BUCKETS,
    CORPORA,
    IGNORE_INDEX,
    Row,
    SkipLog,
    bucket_of,
    cap_by_hours,
    ct2_convert_command,
    decode_audio,
    has_annotator_tag,
    index_corpus,
    index_shard,
    pad_labels,
    plan_split,
    read_checkpoint_step,
    select_shards,
    shard_split,
    speaker_key,
    strip_decoder_start,
    text_for,
    total_hours,
    write_checkpoint_atomically,
)

KATHBATH = CORPORA["kathbath"]
INDICVOICES = CORPORA["indicvoices"]


def make_row(
    corpus: str = "kathbath",
    speaker: str = "kathbath:1",
    seconds: float = 4.0,
    shard: str = "train-00000-of-00001.parquet",
    row_group: int = 0,
    row_in_group: int = 0,
    declared_split: str = "train",
    text: str = "नमस्ते",
) -> Row:
    return Row(
        corpus=corpus,
        shard=shard,
        row_group=row_group,
        row_in_group=row_in_group,
        speaker=speaker,
        seconds=seconds,
        text=text,
        declared_split=declared_split,
    )


class TestTextColumn:
    def test_indicvoices_is_scored_and_trained_on_verbatim(self) -> None:
        # The decision the module docstring argues for, pinned. `text`
        # and `normalized` are byte-identical and both correct the
        # speaker's pronunciation; `verbatim` is what was actually said,
        # and it is what scripts/bench_stt.py already scores against.
        assert INDICVOICES.text_column == "verbatim"
        record = {"text": "दर्शन", "normalized": "दर्शन", "verbatim": "दर्सन"}
        assert text_for(record, INDICVOICES) == "दर्सन"

    def test_kathbath_has_one_transcript_column(self) -> None:
        assert KATHBATH.text_column == "text"
        assert text_for({"text": "नमस्ते"}, KATHBATH) == "नमस्ते"

    def test_a_renamed_column_degrades_to_a_fallback(self) -> None:
        # A schema drift must show up as a different column, not as an
        # empty reference set that reads like "nothing downloaded".
        assert text_for({"text": "योग", "normalized": "योग"}, INDICVOICES) == "योग"

    def test_a_blank_transcript_is_not_a_transcript(self) -> None:
        assert text_for({"verbatim": "   ", "text": ""}, INDICVOICES) == ""
        assert text_for({}, KATHBATH) == ""

    def test_a_blank_preferred_column_falls_through_to_the_fallback(self) -> None:
        # Not the same as the test above: here there IS a transcript, in
        # a column further down the list. Returning "" because the
        # preferred column happened to be whitespace would throw away a
        # usable row and count it as an empty one.
        assert text_for({"verbatim": "  ", "text": "योग"}, INDICVOICES) == "योग"

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert text_for({"text": "  नमस्ते \n"}, KATHBATH) == "नमस्ते"


class TestAnnotatorTags:
    def test_unintelligible_is_a_tag(self) -> None:
        # The only markup that survives IndicVoices' verbatim column.
        # The annotator is saying there is speech here they could not
        # transcribe: training on it teaches the model to emit the
        # literal word.
        assert has_annotator_tag("हाँ मिल जायेगा <unintelligible> मिल जायेगा")

    def test_ordinary_devanagari_is_not(self) -> None:
        assert not has_annotator_tag("योग दर्शन करने से मनुष्य की स्वास्थ्य रेखा अच्छी होती है")

    def test_empty_text_is_not_a_tag(self) -> None:
        assert not has_annotator_tag("")


class TestSpeakerKeys:
    def test_speaker_ids_are_namespaced_by_corpus(self) -> None:
        # Kathbath numbers speakers from 38; a bare integer would collide
        # with anything else that numbers its speakers, and two different
        # people merged into one is the leak the split exists to stop.
        assert speaker_key("kathbath", 252) == "kathbath:252"
        assert speaker_key("indicvoices", "S4259") == "indicvoices:S4259"
        assert speaker_key("kathbath", 1) != speaker_key("indicvoices", 1)

    def test_bucketing_is_the_same_function_the_ser_split_uses(self) -> None:
        # Not `hash()`, which is salted per process and would reshuffle
        # the split on every run.
        assert bucket_of("kathbath:252") == _bucket("kathbath:252", BUCKETS)
        assert 0 <= bucket_of("indicvoices:S1") < BUCKETS

    def test_shard_split_comes_from_the_filename(self) -> None:
        assert shard_split("valid-00000-of-00002.parquet") == "valid"
        assert shard_split(Path("/a/b/train-00007-of-00032.parquet")) == "train"
        assert shard_split("something-else.parquet") == "something"


class TestSkipLog:
    def test_a_typo_in_a_skip_reason_is_an_error(self) -> None:
        # Otherwise the typo silently becomes its own bucket and the
        # count that was supposed to be watched is always zero.
        log = SkipLog()
        with pytest.raises(ValueError, match="unknown skip kind"):
            log.skip("too_lnog")

    def test_counts_add_up(self) -> None:
        log = SkipLog()
        log.skip("empty_text")
        log.skip("empty_text", 3)
        log.skip("too_long")
        assert log.counts["empty_text"] == 4
        assert log.total == 5
        assert log.as_dict()["by_reason"] == {"empty_text": 4, "too_long": 1}

    def test_an_unusable_shard_is_recorded_by_name(self) -> None:
        log = SkipLog()
        log.shard_failed("/data/train-00003.parquet", "OSError mid-read")
        assert log.as_dict()["unusable_shards"] == {"/data/train-00003.parquet": "OSError mid-read"}


class TestIndexShard:
    def _groups(self) -> list[list[dict]]:
        return [
            [
                {"speaker_id": 1, "duration": 4.0, "text": "नमस्ते"},
                {"speaker_id": 2, "duration": 3.0, "text": "शुभ प्रभात"},
            ],
            [{"speaker_id": 1, "duration": 5.0, "text": "धन्यवाद"}],
        ]

    def test_a_row_is_addressed_by_shard_row_group_and_offset(self) -> None:
        log = SkipLog()
        rows = index_shard(
            KATHBATH, "train-00000-of-00001.parquet", self._groups(), log, min_seconds=0.3
        )
        assert [(r.row_group, r.row_in_group) for r in rows] == [(0, 0), (0, 1), (1, 0)]
        assert rows[0].speaker == "kathbath:1" and rows[0].declared_split == "train"
        assert rows[0].text == "नमस्ते"
        assert log.total == 0

    def test_a_clip_longer_than_whispers_window_is_dropped(self) -> None:
        # The feature extractor truncates to 30 s while the transcript
        # keeps the words that were cut off — a label/audio mismatch the
        # loss cannot see.
        log = SkipLog()
        groups = [[{"speaker_id": 1, "duration": 41.0, "text": "बहुत लंबा"}]]
        assert index_shard(KATHBATH, "train-0.parquet", groups, log, min_seconds=0.3) == []
        assert log.counts["too_long"] == 1

    def test_a_clip_below_min_seconds_is_dropped(self) -> None:
        log = SkipLog()
        groups = [[{"speaker_id": 1, "duration": 0.1, "text": "हाँ"}]]
        assert index_shard(KATHBATH, "train-0.parquet", groups, log, min_seconds=0.3) == []
        assert log.counts["too_short"] == 1

    def test_a_blank_transcript_is_counted_not_guessed(self) -> None:
        log = SkipLog()
        groups = [[{"speaker_id": 1, "duration": 4.0, "text": ""}]]
        assert index_shard(KATHBATH, "train-0.parquet", groups, log, min_seconds=0.3) == []
        assert log.counts["empty_text"] == 1

    def test_a_null_duration_is_counted_separately_from_a_missing_column(self) -> None:
        log = SkipLog()
        groups = [
            [{"speaker_id": 1, "duration": None, "text": "अ"}],
            [{"speaker_id": 1, "text": "ब"}],
            [{"duration": 4.0, "text": "स"}],
        ]
        assert index_shard(KATHBATH, "train-0.parquet", groups, log, min_seconds=0.3) == []
        assert log.counts["no_duration"] == 1
        assert log.counts["missing_column"] == 2

    def test_tagged_rows_are_dropped_by_default_and_kept_on_request(self) -> None:
        groups = [[{"speaker_id": 1, "duration": 4.0, "verbatim": "हाँ <unintelligible> ठीक"}]]
        dropped = SkipLog()
        assert index_shard(INDICVOICES, "train-0.parquet", groups, dropped, min_seconds=0.3) == []
        assert dropped.counts["annotator_tag"] == 1

        kept = SkipLog()
        rows = index_shard(
            INDICVOICES, "train-0.parquet", groups, kept, min_seconds=0.3, drop_tagged=False
        )
        assert len(rows) == 1 and kept.total == 0


class TestCapByHours:
    def _pool(self, n: int, corpus: str = "kathbath", seconds: float = 3600.0) -> list[Row]:
        return [
            make_row(corpus=corpus, speaker=f"{corpus}:{i}", seconds=seconds, row_in_group=i)
            for i in range(n)
        ]

    def test_the_budget_is_a_ceiling(self) -> None:
        kept = cap_by_hours(self._pool(20), max_hours=6.0, seed=0)
        assert total_hours(kept) <= 6.0
        assert len(kept) == 6

    def test_no_cap_keeps_everything(self) -> None:
        pool = self._pool(20)
        assert len(cap_by_hours(pool, max_hours=None, seed=0)) == 20

    def test_the_same_seed_picks_the_same_rows(self) -> None:
        pool = self._pool(40)
        first = cap_by_hours(pool, max_hours=5.0, seed=7)
        assert [r.locator for r in first] == [r.locator for r in cap_by_hours(pool, 5.0, 7)]

    def test_a_cap_is_not_the_first_rows_in_read_order(self) -> None:
        # Taking the head of the list would take whole shards, and a
        # shard is a handful of speakers recording in one session — so
        # "six hours of Hindi" would mean "six hours of nine people".
        pool = self._pool(200)
        kept = cap_by_hours(pool, max_hours=10.0, seed=1)
        assert [r.row_in_group for r in kept] != list(range(10))

    def test_rows_come_back_in_read_order(self) -> None:
        kept = cap_by_hours(self._pool(60), max_hours=10.0, seed=3)
        assert [r.locator for r in kept] == sorted(r.locator for r in kept)

    def test_shares_split_the_budget_between_corpora(self) -> None:
        pool = self._pool(30, "kathbath") + self._pool(30, "indicvoices")
        kept = cap_by_hours(pool, 10.0, seed=0, shares={"kathbath": 0.3, "indicvoices": 0.7})
        by_corpus = {c: sum(1 for r in kept if r.corpus == c) for c in ("kathbath", "indicvoices")}
        assert by_corpus == {"kathbath": 3, "indicvoices": 7}

    def test_shares_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            cap_by_hours(self._pool(4), 1.0, 0, shares={"kathbath": 0.3})

    def test_shares_must_name_every_corpus_present(self) -> None:
        pool = self._pool(4, "kathbath") + self._pool(4, "indicvoices")
        with pytest.raises(ValueError, match="missing"):
            cap_by_hours(pool, 1.0, 0, shares={"kathbath": 1.0})


class TestPlanSplit:
    def _corpus(self) -> list[Row]:
        rows = []
        # Kathbath: speakers 1 and 2 appear in BOTH the train shards and
        # the valid shard — which is what the real corpus does.
        for i in range(6):
            rows.append(
                make_row(
                    speaker=f"kathbath:{i % 3}",
                    shard="train-00000-of-00001.parquet",
                    row_in_group=i,
                    declared_split="train",
                )
            )
        for i in range(4):
            rows.append(
                make_row(
                    speaker=f"kathbath:{i % 2}",
                    shard="valid-00000-of-00001.parquet",
                    row_in_group=i,
                    declared_split="valid",
                )
            )
        return rows

    def _indic(self, n: int = 40) -> list[Row]:
        return [
            make_row(
                corpus="indicvoices",
                speaker=f"indicvoices:S{i}",
                shard="train-00000-of-00082.parquet",
                row_in_group=i,
            )
            for i in range(n)
        ]

    def test_kathbath_valid_speakers_are_struck_from_the_training_side(self) -> None:
        # The whole point. Kathbath's `valid` split is the KNOWN-speaker
        # validation set: measured on the real corpus, all 20 of its
        # speakers are also in train, covering 33,863 of 91,752 rows. A
        # recipe that trusted the filename would report a held-out WER
        # for speakers it had just fine-tuned on.
        log = SkipLog()
        plan = plan_split(
            self._corpus(),
            log,
            eval_sets=["kathbath-valid"],
            eval_fraction=0.1,
            eval_limit=None,
            max_train_hours=None,
            seed=0,
        )
        assert plan.evals["kathbath-valid"].speakers == {"kathbath:0", "kathbath:1"}
        assert plan.train.speakers == {"kathbath:2"}
        assert plan.leak_rows == 4
        assert log.counts["speaker_leak"] == 4

    def test_no_speaker_is_ever_on_both_sides(self) -> None:
        log = SkipLog()
        plan = plan_split(
            self._corpus() + self._indic(),
            log,
            eval_fraction=0.2,
            eval_limit=None,
            max_train_hours=None,
            seed=0,
        )
        for partition in plan.evals.values():
            assert not (plan.train.speakers & partition.speakers)

    def test_indicvoices_is_held_out_by_speaker_hash(self) -> None:
        # IndicVoices ships train shards only, so its eval set has to be
        # carved out — by speaker bucket, so the same person lands on the
        # same side on every machine and every run.
        log = SkipLog()
        plan = plan_split(
            self._indic(60),
            log,
            eval_sets=["indicvoices-heldout"],
            eval_fraction=0.25,
            eval_limit=None,
            max_train_hours=None,
            seed=0,
        )
        held = plan.evals["indicvoices-heldout"]
        assert held.rows, "a 25% speaker slice of 60 speakers cannot be empty"
        assert all(bucket_of(s) < 25 for s in held.speakers)
        assert all(bucket_of(s) >= 25 for s in plan.train.speakers)

    def test_the_training_set_does_not_move_when_eval_limit_does(self) -> None:
        # --eval-limit cuts ROWS, and the held-out SPEAKERS are the
        # split. Kathbath is where this bites: a valid-shard speaker also
        # has rows in the train shards, and only the speaker strike
        # removes those. If the struck set came from the SAMPLED rows,
        # then --eval-limit 24 would leave most of those speakers in
        # training (measured on the real corpus: 25,944 rows struck
        # instead of 33,863) — so raising the limit later would quietly
        # move data the adapter was already fitted on.
        rows = self._corpus()
        plans = {
            limit: plan_split(
                rows,
                SkipLog(),
                eval_sets=["kathbath-valid"],
                eval_fraction=0.1,
                eval_limit=limit,
                max_train_hours=None,
                seed=0,
            )
            for limit in (None, 1)
        }
        assert len(plans[1].evals["kathbath-valid"].rows) == 1
        assert len(plans[None].evals["kathbath-valid"].rows) == 4
        assert plans[1].train.speakers == plans[None].train.speakers
        assert [r.locator for r in plans[1].train.rows] == [
            r.locator for r in plans[None].train.rows
        ]

    def test_eval_all_merges_the_sets_in_read_order(self) -> None:
        log = SkipLog()
        plan = plan_split(
            self._corpus() + self._indic(),
            log,
            eval_fraction=0.3,
            eval_limit=None,
            max_train_hours=None,
            seed=0,
        )
        merged = plan.eval_all
        assert len(merged.rows) == sum(len(p.rows) for p in plan.evals.values())
        assert [r.locator for r in merged.rows] == sorted(r.locator for r in merged.rows)

    def test_the_hour_cap_applies_to_training_only(self) -> None:
        log = SkipLog()
        rows = [
            make_row(
                corpus="indicvoices",
                speaker=f"indicvoices:S{i}",
                seconds=3600.0,
                row_in_group=i,
            )
            for i in range(80)
        ]
        plan = plan_split(
            rows,
            log,
            eval_sets=["indicvoices-heldout"],
            eval_fraction=0.25,
            eval_limit=None,
            max_train_hours=3.0,
            seed=0,
        )
        assert plan.train.hours <= 3.0
        assert plan.train_hours_before_cap > 3.0
        assert plan.evals["indicvoices-heldout"].hours > 3.0

    def test_an_unknown_eval_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown eval set"):
            plan_split(
                self._corpus(),
                SkipLog(),
                eval_sets=["kathbath-test"],
                eval_fraction=0.1,
                eval_limit=None,
                max_train_hours=None,
                seed=0,
            )

    def test_a_run_with_no_eval_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one eval set"):
            plan_split(
                self._corpus(),
                SkipLog(),
                eval_sets=[],
                eval_fraction=0.1,
                eval_limit=None,
                max_train_hours=None,
                seed=0,
            )

    def test_the_report_carries_the_leak_count(self) -> None:
        log = SkipLog()
        plan = plan_split(
            self._corpus(),
            log,
            eval_sets=["kathbath-valid"],
            eval_fraction=0.1,
            eval_limit=None,
            max_train_hours=None,
            seed=0,
        )
        summary = plan.as_dict()
        assert summary["speaker_leak_rows_dropped"] == 4
        assert summary["speaker_leak_speakers"] == 2
        assert summary["train"]["by_corpus"] == {"kathbath": 2}


class TestLabels:
    # 50258 is <|startoftranscript|>, which the tokeniser prepends and
    # the model prepends again when it builds decoder_input_ids.
    SOT = 50258

    def test_the_decoder_start_token_is_stripped(self) -> None:
        out = strip_decoder_start([[self.SOT, 1, 2], [self.SOT, 3]], self.SOT)
        assert out == [[1, 2], [3]]

    def test_a_batch_that_does_not_all_start_with_it_is_left_alone(self) -> None:
        # Shifting half a batch by one position is far worse than
        # leaving a token the model will also be given.
        batch = [[self.SOT, 1, 2], [9, 3]]
        assert strip_decoder_start(batch, self.SOT) == batch

    def test_an_empty_batch_is_not_an_error(self) -> None:
        assert strip_decoder_start([], self.SOT) == []

    def test_the_input_is_not_mutated(self) -> None:
        batch = [[self.SOT, 1, 2]]
        strip_decoder_start(batch, self.SOT)
        assert batch == [[self.SOT, 1, 2]]

    def test_labels_are_padded_with_the_ignore_index(self) -> None:
        # Not with the pad token: -100 is what cross-entropy skips, and
        # there is no intermediate state in which a pad token looks like
        # a token to predict.
        assert pad_labels([[1, 2, 3], [4]]) == [[1, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]
        assert IGNORE_INDEX == -100

    def test_padding_an_empty_batch_is_not_an_error(self) -> None:
        assert pad_labels([]) == []


class TestCt2Command:
    def test_the_command_names_the_merged_model_and_the_runtime_directory(self) -> None:
        # A LoRA adapter is invisible to the daemon, which loads
        # CTranslate2 through faster-whisper. A fine-tune that is never
        # converted is a report with no product.
        command = ct2_convert_command("models/merged", "models/out-ct2", "int8_float16")
        assert command[0] == "ct2-transformers-converter"
        assert command[command.index("--model") + 1] == "models/merged"
        assert command[command.index("--output_dir") + 1] == "models/out-ct2"
        assert command[command.index("--quantization") + 1] == "int8_float16"

    def test_the_tokeniser_travels_with_the_converted_model(self) -> None:
        # faster-whisper reads tokenizer.json out of the model directory;
        # without --copy_files the converted model cannot be loaded.
        command = ct2_convert_command("m", "o")
        assert "tokenizer.json" in command and "preprocessor_config.json" in command


class TestDecodeAudio:
    def _flac(self, seconds: float, sr: int, channels: int = 1) -> bytes:
        soundfile = pytest.importorskip("soundfile")
        frames = int(seconds * sr)
        tone = np.sin(np.linspace(0, 2 * math.pi * 220 * seconds, frames)).astype(np.float32)
        data = tone if channels == 1 else np.stack([tone, -tone], axis=1)
        buffer = io.BytesIO()
        soundfile.write(buffer, data, sr, format="FLAC")
        return buffer.getvalue()

    def test_audio_comes_back_as_mono_float32(self) -> None:
        log = SkipLog()
        audio = decode_audio(self._flac(0.5, 16000), log)
        assert audio is not None
        assert audio.dtype == np.float32 and audio.ndim == 1
        assert log.total == 0

    def test_a_stereo_clip_is_mean_down_mixed(self) -> None:
        # The exact path scripts/bench_stt.py and the live provider use,
        # so a clip sounds the same to the trainer as to the recogniser.
        log = SkipLog()
        audio = decode_audio(self._flac(0.25, 16000, channels=2), log)
        assert audio is not None and audio.ndim == 1
        assert np.allclose(audio, 0, atol=1e-3), "L and -R should cancel"

    def test_a_clip_at_another_rate_is_resampled_to_16k(self) -> None:
        pytest.importorskip("librosa")
        log = SkipLog()
        audio = decode_audio(self._flac(1.0, 8000), log)
        assert audio is not None
        assert abs(len(audio) - 16000) < 100, f"expected ~1 s at 16 kHz, got {len(audio)}"

    def test_unreadable_bytes_are_counted_not_raised(self) -> None:
        # One truncated flac somewhere in 63 GB must not end a run that
        # has been going for an hour.
        log = SkipLog()
        assert decode_audio(b"not audio at all", log) is None
        assert log.counts["decode_failed"] == 1

    def test_a_row_with_no_audio_is_its_own_reason(self) -> None:
        log = SkipLog()
        assert decode_audio(None, log) is None
        assert log.counts["no_audio"] == 1


class TestParquetIndex:
    """Against tiny synthetic shards, never the 63 GB on disk.

    pyarrow lives in the training venv (`scripts/bench_stt.py` shells out
    to it for the same reason), so these skip in the runtime venv. Run
    them with `training/.venv/bin/python -m pytest`.
    """

    @pytest.fixture(autouse=True)
    def _needs_pyarrow(self) -> None:
        pytest.importorskip("pyarrow")

    def _write(self, path: Path, rows: list[dict], row_group_size: int = 2) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        pa = pytest.importorskip("pyarrow")
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, row_group_size=row_group_size)

    def _corpus(self, tmp_path: Path) -> Path:
        root = tmp_path / "kathbath"
        (root / ".elizabeth-complete").parent.mkdir(parents=True, exist_ok=True)
        (root / ".elizabeth-complete").write_text("test\n")
        self._write(
            root / "hindi" / "train-00000-of-00001.parquet",
            [
                {"speaker_id": 1, "duration": 4.0, "text": "नमस्ते"},
                {"speaker_id": 2, "duration": 3.0, "text": "शुभ प्रभात"},
                {"speaker_id": 1, "duration": 2.0, "text": "धन्यवाद"},
            ],
        )
        self._write(
            root / "hindi" / "valid-00000-of-00001.parquet",
            [{"speaker_id": 1, "duration": 5.0, "text": "फिर मिलेंगे"}],
        )
        return tmp_path

    def test_rows_are_indexed_with_their_row_group(self, tmp_path: Path) -> None:
        data_root = self._corpus(tmp_path)
        log = SkipLog()
        rows = index_corpus(KATHBATH, data_root, log, min_seconds=0.3)
        assert len(rows) == 4
        train = [r for r in rows if r.declared_split == "train"]
        # row_group_size=2 over 3 rows: two groups, offsets restart at 0.
        assert [(r.row_group, r.row_in_group) for r in train] == [(0, 0), (0, 1), (1, 0)]
        assert {r.speaker for r in rows} == {"kathbath:1", "kathbath:2"}
        assert log.total == 0

    def test_the_held_out_shard_is_recognised_by_its_filename(self, tmp_path: Path) -> None:
        rows = index_corpus(KATHBATH, self._corpus(tmp_path), SkipLog(), min_seconds=0.3)
        assert [r.declared_split for r in rows if r.text == "फिर मिलेंगे"] == ["valid"]

    def test_a_missing_corpus_is_a_note_not_a_crash(self, tmp_path: Path) -> None:
        log = SkipLog()
        assert index_corpus(KATHBATH, tmp_path, log, min_seconds=0.3) == []
        assert any("not downloaded" in r for r in log.shards.values())

    def test_a_shard_without_a_transcript_column_is_named(self, tmp_path: Path) -> None:
        root = tmp_path / "kathbath" / "hindi"
        self._write(root / "train-00000-of-00001.parquet", [{"speaker_id": 1, "duration": 4.0}])
        log = SkipLog()
        assert index_corpus(KATHBATH, tmp_path, log, min_seconds=0.3) == []
        assert any("no transcript column" in r for r in log.shards.values())

    def test_a_corrupt_shard_is_skipped_and_the_rest_are_read(self, tmp_path: Path) -> None:
        data_root = self._corpus(tmp_path)
        (data_root / "kathbath" / "hindi" / "train-00001-of-00002.parquet").write_bytes(b"PAR1junk")
        log = SkipLog()
        rows = index_corpus(KATHBATH, data_root, log, min_seconds=0.3)
        assert len(rows) == 4, "the good shards still produced their rows"
        assert len(log.shards) == 1

    def test_hidden_cache_files_are_not_read_as_shards(self, tmp_path: Path) -> None:
        # HF's .cache/ keeps lock and partial files next to real shards,
        # and a .parquet.incomplete is exactly what must not be read.
        data_root = self._corpus(tmp_path)
        hidden = data_root / "kathbath" / "hindi" / ".cache" / "train-00009-of-00009.parquet"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        hidden.write_bytes(b"PAR1junk")
        found = select_shards(data_root / "kathbath" / "hindi")
        assert all(".cache" not in str(p) for p in found)
        assert len(found) == 2


class TestCheckpointing:
    """A long run on a machine with a flaky power supply — the reason
    these exist. Verified live once too: a real training pass on
    Kathbath saved at step 3 and 6, and a resumed run picked up at 6
    and finished cleanly (2026-09-14) — this class covers the atomicity
    and step-recording pieces that don't need a real model to test.
    """

    def test_a_cut_mid_save_leaves_the_previous_checkpoint_intact(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint"

        def first_save(dest: Path) -> None:
            dest.mkdir(parents=True)
            (dest / "adapter.bin").write_text("good weights at step 100")

        write_checkpoint_atomically(first_save, ckpt, 100)
        assert (ckpt / "adapter.bin").read_text() == "good weights at step 100"

        def cut_mid_save(dest: Path) -> None:
            dest.mkdir(parents=True)
            (dest / "adapter.bin").write_text("half-written")
            raise OSError("power cut")

        with pytest.raises(OSError, match="power cut"):
            write_checkpoint_atomically(cut_mid_save, ckpt, 200)

        # The live checkpoint is exactly what it was before the failed
        # save — not the half-written attempt, not gone entirely.
        assert (ckpt / "adapter.bin").read_text() == "good weights at step 100"
        assert read_checkpoint_step(ckpt) == 100

    def test_step_is_recorded_and_read_back(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint"
        write_checkpoint_atomically(lambda dest: dest.mkdir(parents=True), ckpt, 4200)
        assert read_checkpoint_step(ckpt) == 4200

    def test_a_checkpoint_with_no_step_file_resumes_from_zero_not_an_error(
        self, tmp_path: Path
    ) -> None:
        # An older save, or one from a version of this format that
        # predates step.json — safe to under-count, never to crash.
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        assert read_checkpoint_step(ckpt) == 0

    def test_repeated_saves_do_not_accumulate_directories(self, tmp_path: Path) -> None:
        # One checkpoint to lose in a power cut, not thirty filling the
        # disk over a 20-hour run.
        ckpt = tmp_path / "checkpoint"
        for step in (100, 200, 300):
            write_checkpoint_atomically(
                lambda dest, s=step: (dest.mkdir(parents=True), (dest / f"s{s}").write_text("x")),
                ckpt,
                step,
            )
        assert read_checkpoint_step(ckpt) == 300
        assert {p.name for p in ckpt.iterdir()} == {"s300", "step.json"}
        assert not (tmp_path / "checkpoint.tmp").exists()
