"""Always-on "Hey Elizabeth" — the thing that presses the spacebar.

`elizabeth talk` starts an utterance when SPACE is pressed, and being
asked to reach for a key before speaking is the single complaint that
started this module. `listen.py` already knows how to *end* an utterance
without the key; this is the other half, and it is the one that decides
whether a microphone that is always open ever wakes anything up.

**The pipeline, all three shapes measured off the graphs rather than
remembered.**

    1280 samples of 16 kHz audio (80 ms)
      -> melspectrogram.onnx        -> 5 mel frames of 32 bins
      -> mel / 10 + 2               -> what the embedder was trained on
      -> embedding_model.onnx       -> one 96-d vector per 76 mel frames
      -> head.onnx                  -> one score per 16 embeddings

Sixteen embeddings at an 80 ms stride is 1.28 s of context, comfortably
longer than the phrase. Cost on this laptop's CPU: 0.18 ms of mel and
1.75 ms of embedding per 80 ms hop — about 2.4% of one core to listen
for your name forever. That measurement is why nothing gates this behind
the VAD: gating would save two percent of a core and add a way for the
first syllable of the phrase to be the one that arms the gate.

**What is third-party and what is ours.** The two feature extractors are
openWakeWord's, Apache-2.0, pulled from `littlebearlabs/openwakeword-
features` — not from the author's own Hub mirror, which is
cc-by-nc-sa-4.0 and carries no weights anyway, and not from a GitHub
release, which this project's network blocks (CLAUDE.md rule 3). The
head is trained here, on speech this project generated. The
`openwakeword` PyPI package is deliberately NOT a dependency: on Linux
it requires `tflite-runtime`, whose wheels stop at cp311 while this
project is 3.12, so it cannot install without `--no-deps`. Two ONNX
files and onnxruntime do the same work with nothing to work around.

**The failure this module is shaped to avoid.** Every step here is a
silent one. Feed the embedder un-scaled mel, or 512-sample frames
because that is what Silero wants, or frames out of order, and nothing
raises — the score simply never crosses the threshold, which looks
exactly like "the wake word does not work" and not at all like a bug.
So the frame size is enforced by buffering rather than assumed, the
scaling is named in config with a comment saying what skipping it looks
like, and `WakeWord.diagnostics()` reports the live score so "is it
hearing me at all" is answerable without a debugger.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from elizabeth.config import Elizabeth, WakeConfig

# The head's two-dimensional output is (batch, 1). Named so the squeeze
# below reads as a decision rather than an incantation.
_SCORE_AXES = (0, 1)


class WakeModelMissing(FileNotFoundError):
    """The head has not been trained yet, or the features were never
    fetched. Raised with the command that fixes it, because the
    alternative — a detector that loads and never fires — is the single
    most expensive failure mode this module has.
    """


@dataclass(frozen=True)
class WakeEvent:
    """She heard her name.

    `score` is the head's probability at the hop that crossed, and
    `t_detected` is the monotonic clock at that hop. The phrase ENDED
    around `t_detected`; it started roughly `context_s` earlier, which is
    what the caller needs to know to pull the right audio out of the
    ring if it wants the words that followed the name in the same breath.
    """

    score: float
    t_detected: float
    context_s: float


class WakeWord:
    """Feed it audio, it tells you when it heard the phrase.

    Frames may be any length: audio is accumulated and consumed in exact
    `hop_samples` chunks, so the caller can hand over the same 512-sample
    blocks `listen.py` takes without either of them needing to know about
    the other's window size. Frames must still arrive IN ORDER — the mel
    and embedding buffers are a sliding context, and a skipped block is a
    gap in the phrase.

    Audio is float32 in [-1, 1], this project's convention everywhere
    else. The ONNX front end expects int16-range values, and the scaling
    happens here rather than at each call site.
    """

    def __init__(self, cfg: WakeConfig, *, head: object | None = None,
                 features: tuple[object, object] | None = None) -> None:
        self.cfg = cfg
        self._pending = np.zeros(0, dtype=np.float32)
        self._mel: deque[np.ndarray] = deque(maxlen=cfg.mel_window)
        self._emb: deque[np.ndarray] = deque(maxlen=cfg.embedding_window)
        self._muted_until = 0.0
        self._last_score = 0.0
        self._hops = 0

        # Injected sessions keep every test off the disk and off onnxruntime;
        # the real ones are loaded only when nothing was supplied.
        if features is None or head is None:
            loaded_mel, loaded_emb, loaded_head = _load_sessions(cfg)
            self._mel_sess = features[0] if features else loaded_mel
            self._emb_sess = features[1] if features else loaded_emb
            self._head = head or loaded_head
        else:
            self._mel_sess, self._emb_sess = features
            self._head = head

        # The context the classifier sees before any real audio arrives is
        # the mel of silence, not zeros. They are not the same thing after
        # `mel / 10 + 2`, and a classifier primed with impossible values
        # spends its first second of every run in a state it never saw in
        # training.
        self._prime_with_silence()

    # -- construction ------------------------------------------------------

    @classmethod
    def with_models(cls, cfg: Elizabeth | None = None) -> WakeWord:
        return cls((cfg or Elizabeth()).wake)

    # -- the loop ----------------------------------------------------------

    def feed(self, frame: np.ndarray, *, now: float) -> WakeEvent | None:
        """Consume audio; return a `WakeEvent` on the hop that crossed.

        `now` is passed in rather than read from a clock so the caller's
        timestamps and this one's agree, and so tests are not timing
        tests.
        """
        if not self.cfg.enabled:
            return None

        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate((self._pending, frame))

        event: WakeEvent | None = None
        hop = self.cfg.hop_samples
        while self._pending.size >= hop:
            chunk, self._pending = self._pending[:hop], self._pending[hop:]
            score = self._score_hop(chunk)
            if score is None:
                continue
            self._last_score = score
            self._hops += 1
            if now < self._muted_until or score < self.cfg.threshold:
                continue
            # Refractory FIRST, and the context cleared with it: one
            # spoken phrase crosses the threshold on several consecutive
            # hops, and without this she answers her own name three times.
            self._muted_until = now + self.cfg.refractory_ms / 1000.0
            self._emb.clear()
            event = WakeEvent(score=score, t_detected=now, context_s=self.context_s)
        return event

    def reset(self) -> None:
        """Forget the context. For after a barge-in, a device change, or
        anything else that makes the last second of audio a lie.
        """
        self._pending = np.zeros(0, dtype=np.float32)
        self._emb.clear()
        self._muted_until = 0.0
        self._prime_with_silence()

    # -- introspection -----------------------------------------------------

    @property
    def context_s(self) -> float:
        """Seconds of audio the head scores at once."""
        return self.cfg.embedding_window * self.cfg.hop_samples / 16000.0

    def diagnostics(self) -> dict[str, float]:
        """Enough to answer "is it hearing me at all" without a debugger.

        `last_score` against `threshold` is the whole question: a score
        that moves with your voice but tops out at 0.3 is a threshold
        problem, and one pinned at 0.0 while you shout is a pipeline
        problem. They look identical from the outside and have nothing
        in common as fixes.
        """
        return {
            "last_score": self._last_score,
            "threshold": self.cfg.threshold,
            "hops_scored": float(self._hops),
            "context_s": self.context_s,
            "embeddings_buffered": float(len(self._emb)),
        }

    # -- internals ---------------------------------------------------------

    def _score_hop(self, chunk: np.ndarray) -> float | None:
        """One hop through the whole pipeline. None until the buffers fill."""
        # int16 range: what the ONNX front end was traced on. Our audio is
        # float [-1, 1] everywhere else, so the conversion lives here and
        # not at five call sites.
        pcm = (chunk * 32767.0).astype(np.float32).reshape(1, -1)
        mel = self._mel_sess.run(None, {"input": pcm})[0]
        mel = mel.squeeze()  # (frames, 32)
        mel = mel / self.cfg.mel_scale + self.cfg.mel_offset
        for row in np.atleast_2d(mel):
            self._mel.append(row.astype(np.float32))

        if len(self._mel) < self.cfg.mel_window:
            return None
        window = np.stack(self._mel)[None, :, :, None]  # (1, 76, 32, 1)
        emb = self._emb_sess.run(None, {"input_1": window.astype(np.float32)})[0]
        self._emb.append(emb.reshape(-1))

        if len(self._emb) < self.cfg.embedding_window:
            return None
        stack = np.stack(self._emb)[None, :, :].astype(np.float32)  # (1, 16, 96)
        name = self._head.get_inputs()[0].name
        out = self._head.run(None, {name: stack})[0]
        return float(np.asarray(out).squeeze())

    def _prime_with_silence(self) -> None:
        self._mel.clear()
        silence = np.zeros(self.cfg.hop_samples, dtype=np.float32)
        # Enough hops to fill the mel window whatever the frames-per-hop
        # ratio turns out to be; the deque's maxlen does the trimming.
        for _ in range(self.cfg.mel_window):
            if len(self._mel) >= self.cfg.mel_window:
                break
            pcm = silence.reshape(1, -1)
            mel = self._mel_sess.run(None, {"input": pcm})[0].squeeze()
            mel = mel / self.cfg.mel_scale + self.cfg.mel_offset
            for row in np.atleast_2d(mel):
                self._mel.append(row.astype(np.float32))


def _load_sessions(cfg: WakeConfig) -> tuple[object, object, object]:
    import onnxruntime as ort

    features = Path(cfg.features_dir)
    mel_path = features / "melspectrogram.onnx"
    emb_path = features / "embedding_model.onnx"
    head_path = Path(cfg.head_path)

    missing = [p for p in (mel_path, emb_path) if not p.exists()]
    if missing:
        raise WakeModelMissing(
            f"wake feature extractors missing: {', '.join(str(p) for p in missing)}. "
            "Run: uv run python scripts/fetch_models.py --component wake"
        )
    if not head_path.exists():
        raise WakeModelMissing(
            f"no wake head at {head_path} — the phrase {cfg.phrase!r} has not been "
            "trained yet. Run: uv run python scripts/train_wake.py"
        )

    # One thread each. This runs forever alongside a trainer that already
    # wants every core; a front end that spawns a pool per session turns
    # 2% of a core into a load-average problem.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(mel_path), opts, providers=providers),
        ort.InferenceSession(str(emb_path), opts, providers=providers),
        ort.InferenceSession(str(head_path), opts, providers=providers),
    )
