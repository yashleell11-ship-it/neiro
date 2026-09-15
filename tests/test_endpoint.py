"""smart-turn: has he finished a turn, or just paused?

Calibrated against pipecat's own labelled data rather than intuition —
200 endpoints and 200 non-endpoints, AUC 0.996.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elizabeth.audio.endpoint import EndpointerUnavailable, SmartTurnEndpointer, log_mel
from elizabeth.config import Elizabeth

MODEL = Path(__file__).resolve().parents[1] / "models/smart-turn-v3/smart-turn-v3.2-cpu.onnx"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="smart-turn not downloaded")


class TestFeatures:
    def test_the_shape_is_exactly_what_the_onnx_wants(self) -> None:
        cfg = Elizabeth()
        for seconds in (0.5, 3.0, 8.0, 20.0):
            pcm = np.zeros(int(seconds * 16000), dtype=np.float32)
            assert log_mel(pcm, cfg).shape == (1, cfg.endpoint.n_mels, cfg.endpoint.n_frames)

    def test_long_audio_keeps_the_MOST_RECENT_window(self) -> None:
        # The END of an utterance is what decides whether it ended, so
        # truncating from the front would throw away the only part that
        # matters.
        cfg = Elizabeth()
        loud_tail = np.concatenate(
            [np.zeros(16000 * 20, dtype=np.float32), np.ones(16000 * 2, dtype=np.float32) * 0.5]
        )
        loud_head = np.concatenate(
            [np.ones(16000 * 2, dtype=np.float32) * 0.5, np.zeros(16000 * 20, dtype=np.float32)]
        )
        assert log_mel(loud_tail, cfg).mean() != pytest.approx(log_mel(loud_head, cfg).mean())

    def test_short_audio_is_padded_on_the_LEFT(self) -> None:
        # So the real speech still sits at the right-hand edge, where the
        # model expects the end of a turn to be.
        cfg = Elizabeth()
        pcm = np.ones(16000, dtype=np.float32) * 0.5
        mel = log_mel(pcm, cfg)[0]
        assert mel[:, -1].mean() > mel[:, 0].mean()


class TestThresholdPolicy:
    def test_it_favours_letting_him_finish(self) -> None:
        # Cutting someone off mid-sentence costs the whole turn; being a
        # moment slow costs a moment. Measured: 0.70 halves false cuts
        # against 0.60 for one point of recall.
        cfg = Elizabeth()
        assert cfg.endpoint.complete_threshold >= 0.65

    def test_there_is_a_hard_ceiling(self) -> None:
        # A model that never fires must not be able to hang the
        # conversation.
        cfg = Elizabeth()
        assert 1.0 < cfg.endpoint.max_wait_s <= 4.0

    def test_max_wait_ends_the_turn_regardless(self) -> None:
        e = SmartTurnEndpointer()
        complete, p = e.is_complete(np.zeros(1600, dtype=np.float32), waited_s=99.0)
        assert complete and p == 1.0

    def test_a_missing_model_says_how_to_fix_it(self) -> None:
        e = SmartTurnEndpointer(model_path=Path("/nonexistent/smart-turn.onnx"))
        with pytest.raises(EndpointerUnavailable, match="fetch-models"):
            e.load()


@pytest.fixture(scope="module")
def endpointer() -> SmartTurnEndpointer:
    e = SmartTurnEndpointer()
    e.warm()
    return e


@needs_model
class TestAgainstTheModel:
    def test_a_probability_is_a_probability(self, endpointer: SmartTurnEndpointer) -> None:
        p = endpointer.probability(np.zeros(16000 * 2, dtype=np.float32))
        assert 0.0 <= p <= 1.0

    def test_a_complete_utterance_scores_above_an_abrupt_cut(
        self, endpointer: SmartTurnEndpointer
    ) -> None:
        # The discrimination the whole component exists for. Measured on
        # pipecat's labelled set at AUC 0.996; this is the cheap version
        # that runs without the corpus.
        corpus = Path(__file__).resolve().parents[1] / "data/datasets/crema-d/extracted"
        clips = sorted(corpus.rglob("1001_*.wav"))[:1] if corpus.is_dir() else []
        if not clips:
            pytest.skip("CREMA-D not extracted")
        import soundfile as sf

        audio, _ = sf.read(clips[0], dtype="float32")
        whole = endpointer.probability(audio)
        cut = endpointer.probability(audio[: int(len(audio) * 0.6)])
        assert whole > cut

    def test_it_is_fast_enough_to_run_per_candidate_endpoint(
        self, endpointer: SmartTurnEndpointer
    ) -> None:
        import time

        audio = np.zeros(16000 * 3, dtype=np.float32)
        endpointer.probability(audio)
        started = time.perf_counter()
        for _ in range(3):
            endpointer.probability(audio)
        assert (time.perf_counter() - started) / 3 < 0.2
