"""faster-whisper implementation of protocols.STT.

Model pick corrects the spec: distil-whisper/distil-large-v3.5-ct2, not
whisper-small — it scores 3.60 WER on the Open ASR Leaderboard's
Indian-accented-English column against 3.95 for full whisper-large-v3
(whisper-small isn't even on that column), is MIT, and costs ~0.8 GB in
int8. Confirmed working on this exact GPU in Gate G1 (docs/DECISIONS.md).

The call flags matter more than the model choice: `beam_size=1` (not
the default 5) and `condition_on_previous_text=False` are the two
biggest latency and anti-hallucination wins on short (2-5s) clips.

The confidence floor and hallucination blocklist below matter MORE than
either. Whisper hallucinates confidently on silence and near-silence,
and its single most common hallucination ("thank you", almost certainly
from YouTube-caption training data) reads exactly like a real,
well-formed sentence — nothing about the text itself distinguishes it
from something actually said. The model's own reported confidence
(avg_logprob, no_speech_prob) is the only signal available, so this
provider filters on it before the caller ever sees the text: an empty
string return means "nothing usable was said," not an error.

Language comes from `cfg.stt.language`, never a literal here. "auto"
hands faster-whisper `language=None`, so it detects per utterance —
Hindi, English and Hinglish in one sitting is the normal case for this
user, not an edge case. The code it detected is kept on
`last_detected_language` so the benchmark can print the detector's
histogram beside the WER: "Hindi audio scored badly" and "Hindi audio
was tagged as Urdu" are different problems with different fixes, and a
WER alone cannot tell them apart.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from neiro.config import Neiro
from neiro.state import Locality

# Whisper's well-documented non-speech hallucinations. Exact match,
# case-insensitive, after stripping trailing punctuation/whitespace.
_HALLUCINATION_BLOCKLIST = {
    "",
    "you",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching!",
    "bye",
    "bye bye",
}

_LD_LIBRARY_PATH_HINT = (
    "libcublas.so.12 not found — LD_LIBRARY_PATH wasn't set before THIS "
    "process started. Run:\n\n    source env.sh\n\nin this shell, then "
    "re-run. Setting it from inside Python after the process has already "
    "started does NOT work — confirmed empirically, see docs/DECISIONS.md."
)


class FasterWhisperStt:
    """protocols.STT implementation. `transcribe()` is lazy: the model
    loads (and warms) on first call unless `warm()` was called explicitly
    first — call `warm()` at process start in any long-running process
    so a real user utterance never pays the first-use JIT cost.
    """

    locality = Locality.LAN_TIERABLE

    def __init__(self, cfg: Neiro | None = None) -> None:
        self._cfg = cfg or Neiro()
        self._model = None
        # ISO code faster-whisper settled on for the last utterance —
        # detected under "auto", echoed back when pinned. Diagnostic
        # only: the orchestrator reads the text, never this.
        self.last_detected_language: str | None = None

    def warm(self) -> float:
        """Load the model AND run one throwaway transcribe.

        Gate G1 measured the real first-use stall as ~7.4s on
        *transcribe*, not on model construction (which only cost ~3s
        cold vs 1.67s warm — an unremarkable delta). Warming with only
        `WhisperModel(...)` and no inference call would still leave the
        7.4s stall sitting in front of the first real sentence.
        """
        t0 = time.perf_counter()
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._cfg.stt.model_id,
                device="cuda",
                compute_type=self._cfg.stt.compute_type,
            )
            silence = np.zeros(self._cfg.audio.input_samplerate, dtype="float32")
            segments, _info = self._model.transcribe(silence, beam_size=1, without_timestamps=True)
            list(segments)  # force the generator to actually run
        except RuntimeError as exc:
            if "libcublas" in str(exc):
                raise RuntimeError(_LD_LIBRARY_PATH_HINT) from exc
            raise
        return time.perf_counter() - t0

    async def transcribe(self, pcm_16k: np.ndarray) -> str:
        # ctranslate2 is synchronous and holds the GIL for most of the
        # 150-400 ms it spends decoding. Inline, that freezes the event
        # loop that is also serving the browser socket, the rolling
        # affect window and the cancel check — "cancel is checked at
        # every await" is only true if the awaits can actually run.
        # Same shape as KokoroTts and MoonshineStt.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_16k)

    def _transcribe_sync(self, pcm_16k: np.ndarray) -> str:
        if self._model is None:
            self.warm()

        # None is faster-whisper's "detect it" — one extra decoder pass
        # over the first window, paid only under "auto".
        language = None if self._cfg.stt.language == "auto" else self._cfg.stt.language
        segments, info = self._model.transcribe(
            pcm_16k,
            beam_size=self._cfg.stt.beam_size,
            condition_on_previous_text=self._cfg.stt.condition_on_previous_text,
            vad_filter=False,
            language=language,
            without_timestamps=True,
        )
        self.last_detected_language = getattr(info, "language", None)
        segments = list(segments)
        if not segments:
            return ""

        # faster-whisper reports these per segment; take the worst value
        # across the utterance — one bad segment is enough to distrust
        # the whole turn.
        worst_avg_logprob = min(s.avg_logprob for s in segments)
        worst_no_speech = max(s.no_speech_prob for s in segments)
        text = "".join(s.text for s in segments).strip()

        if worst_no_speech > self._cfg.stt.no_speech_prob_floor:
            return ""
        if worst_avg_logprob < self._cfg.stt.avg_logprob_floor:
            return ""
        if text.strip(".,!? ").lower() in _HALLUCINATION_BLOCKLIST:
            return ""

        return text
