#!/usr/bin/env python3
"""Teach the head to hear "Hey Elizabeth", and say how often it is wrong.

    uv run --extra wake-train python scripts/train_wake.py

Nothing here touches a GPU. Both cards are pinned by the persona and STT
lanes for days at a time, and a job that waits for a free GPU on a
machine that trains around the clock never runs at all. The head is
~200k parameters over a 1536-d input; on CPU it fits in a couple of
minutes, and the expensive half is the ONNX front end, which is CPU-only
anyway.

**Where the data comes from, and why none of it is recorded.** Positives
are this project's own Kokoro voice saying the phrase across every
English voice it ships, at several speeds. Negatives are two kinds, and
both kinds matter: *confusables*, the same voices saying things that
share most of the phrase ("Hey Eliza", "Elizabeth", "Hey Alexa", "say
Elizabeth"), and *bulk speech*, real humans from LibriSpeech who are
talking about something else entirely. Training on bulk speech alone
produces a head that fires on any two-syllable-then-four-syllable
utterance; training on confusables alone produces one that has never
heard a room.

**The metric, and only this one** (CLAUDE.md rule 8). A wake word has an
asymmetric cost: a miss makes you say it twice, a false accept makes her
start talking while you are on a call. So the reported number is **false
accepts per hour on held-out LibriSpeech** at the recall the threshold
buys, swept across thresholds. Accuracy is not reported because at the
natural class balance it would read 99.9% for a head that never fires.

**The augmentation is not decoration.** A head trained on clean TTS and
nothing else scores 1.0 on clean TTS and falls apart the first time a
fan is on. Each positive is rendered once and then mixed at several
SNRs against real speech babble and against noise, with random gain and
random alignment inside the window — which also teaches it that the
phrase can end anywhere in the 1.28 s the classifier sees.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from elizabeth.config import Elizabeth, WakeConfig

SAMPLE_RATE = 16000
KOKORO_RATE = 24000

# Every English voice Kokoro ships: American female/male, British
# female/male. The other 26 are Spanish, French, Hindi, Italian,
# Japanese, Portuguese and Mandarin — a wake word trained on a Mandarin
# speaker saying an English name learns the wrong thing.
ENGLISH_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")

# The phrase, written the several ways a TTS front end will pronounce
# differently. "Hey, Elizabeth" with the comma gets a real pause; the
# bare one runs together. Both are things a person says.
POSITIVE_TEXTS = (
    "Hey Elizabeth.",
    "Hey, Elizabeth.",
    "Hey Elizabeth!",
    "Hey Elizabeth?",
    "hey elizabeth",
)

# Confusables. Each one shares a chunk of the phrase, which is exactly
# what a head trained only against unrelated speech will happily accept.
NEGATIVE_TEXTS = (
    "Hey.", "Elizabeth.", "Hey Eliza.", "Hey Lizzy.", "Hey Beth.",
    "Hello Elizabeth.", "Say Elizabeth.", "Elizabeth, hey.", "Hey Alexa.",
    "Hey Google.", "Hey Jarvis.", "Hey buddy.", "Elizabeth Bennett.",
    "Hey Elizabeth's brother said so.", "They let his birthday pass.",
    "Are you Elizabeth?", "It's Elizabeth on the phone.", "Hey there.",
    "A elizabeth.", "Heyyy.", "Elizabethan.", "Hey it's me.",
)


def english_voices(models_dir: Path) -> list[str]:
    voices = sorted(p.stem for p in (models_dir / "kokoro-82m" / "voices").glob("*.pt"))
    return [v for v in voices if v.startswith(ENGLISH_VOICE_PREFIXES)]


def to_16k(pcm: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(pcm, 2, 3).astype(np.float32)  # 24000 -> 16000


def synthesise(texts, voices, speeds, cache: Path) -> list[Path]:
    """Render every (text, voice, speed) once, to disk.

    Cached because Kokoro is the slowest thing in this script by an order
    of magnitude and the rest of the pipeline gets re-run while tuning.
    """
    import soundfile as sf

    from elizabeth.tts.kokoro import KokoroTts

    cache.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    todo = [(t, v, s) for v in voices for t in texts for s in speeds]
    for i, (text, voice, speed) in enumerate(todo, 1):
        stem = f"{voice}__{abs(hash(text)) % 10**8}__{speed:.2f}".replace(".", "_")
        path = cache / f"{stem}.wav"
        out.append(path)
        if path.exists():
            continue
        tts = KokoroTts(voice=voice)
        pipeline = tts._load()
        pcm = np.concatenate(
            [np.asarray(r.audio, dtype=np.float32).reshape(-1)
             for r in pipeline(text, voice=voice, speed=speed)
             if getattr(r, "audio", None) is not None]
        )
        sf.write(path, to_16k(pcm), SAMPLE_RATE)
        if i % 25 == 0:
            print(f"  synth {i}/{len(todo)}", flush=True)
    return out


def librispeech_clips(root: Path, shards: int, per_shard: int, rng: random.Random):
    """Real human speech, decoded straight out of the parquet shards.

    LibriSpeech ships as parquet with the flac bytes inline, so nothing
    is extracted to disk — 116 GB stays 116 GB and this reads the few
    hundred clips it wants.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    files = sorted(root.glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet shards under {root}")
    rng.shuffle(files)
    for f in files[:shards]:
        table = pq.read_table(f, columns=["audio"])
        rows = table.column("audio").to_pylist()
        rng.shuffle(rows)
        for row in rows[:per_shard]:
            raw = row.get("bytes") if isinstance(row, dict) else None
            if not raw:
                continue
            pcm, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa

                pcm = librosa.resample(pcm, orig_sr=sr, target_sr=SAMPLE_RATE)
            yield pcm.astype(np.float32)


class Embedder:
    """Audio in, a sequence of 96-d vectors out — the same front end the
    detector runs, loaded once and reused.

    Deliberately not `WakeWord` itself: that class is a detector with a
    threshold, a refractory window and a head. Here we want every window
    it would ever score, labelled, with no head in the way.
    """

    def __init__(self, cfg: WakeConfig) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        d = Path(cfg.features_dir)
        self.cfg = cfg
        self.mel = ort.InferenceSession(str(d / "melspectrogram.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        self.emb = ort.InferenceSession(str(d / "embedding_model.onnx"), opts,
                                        providers=["CPUExecutionProvider"])

        # Measured once, from the graph, rather than assumed from the 80 ms
        # hop: how many mel frames one hop of audio produces, and therefore
        # how many samples one mel frame is worth. Everything about WHERE the
        # phrase sits inside a scored window depends on this number.
        probe = np.zeros(cfg.hop_samples * 16, dtype=np.float32).reshape(1, -1)
        frames = np.atleast_2d(self.mel.run(None, {"input": probe})[0].squeeze()).shape[0]
        self.frames_per_hop = max(1, round(frames / 16))
        self.mel_hop_samples = cfg.hop_samples // self.frames_per_hop

        # THE number that made the first run produce zero positive examples.
        # A scored window is 16 embeddings at an 80 ms stride -- but each
        # embedding summarises 76 mel frames, about 760 ms, so the audio the
        # head actually sees is 1.28 s of stride PLUS that 760 ms tail of
        # history: ~2.0 s, not 1.28 s. Labelling positives with the 1.28 s
        # figure put every label 0.68 s behind the phrase, which is to say on
        # background, which is to say there were no positives at all and the
        # sweep reported recall as nan.
        self.receptive_samples = (
            (cfg.embedding_window - 1) * cfg.hop_samples
            + cfg.mel_window * self.mel_hop_samples
        )

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        """(n_hops, 96). One vector per 80 ms, each summarising the 760 ms
        of mel that ended there.
        """
        cfg = self.cfg
        hop = cfg.hop_samples
        usable = (len(audio) // hop) * hop
        if usable < hop:
            return np.zeros((0, 96), dtype=np.float32)
        pcm = (audio[:usable] * 32767.0).astype(np.float32).reshape(1, -1)
        mel = self.mel.run(None, {"input": pcm})[0].squeeze()
        mel = mel / cfg.mel_scale + cfg.mel_offset
        mel = np.atleast_2d(mel).astype(np.float32)

        out = []
        for end in range(cfg.mel_window, mel.shape[0] + 1, self.frames_per_hop):
            window = mel[end - cfg.mel_window:end][None, :, :, None]
            out.append(self.emb.run(None, {"input_1": window})[0].reshape(-1))
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 96), np.float32)

    def windows(self, audio: np.ndarray) -> np.ndarray:
        """(n, 16, 96) — every context the head would ever be handed."""
        e = self.embeddings(audio)
        n = self.cfg.embedding_window
        if len(e) < n:
            return np.zeros((0, n, 96), dtype=np.float32)
        return np.stack([e[i:i + n] for i in range(len(e) - n + 1)])

    def window_span(self, i: int) -> tuple[int, int]:
        """(first sample, last sample) of the audio window `i` was built from."""
        start = i * self.cfg.hop_samples
        return start, start + self.receptive_samples


def augment(phrase: np.ndarray, background: np.ndarray, rng: random.Random,
            cfg: WakeConfig, receptive: int) -> tuple[np.ndarray, int, int]:
    """One example: the phrase placed so a real detector could catch it.

    Returns the clip and the phrase's (start, end) in samples, because
    which windows count as positive depends on where the phrase actually
    landed — labelling every window of a clip that contains the phrase
    would teach the head that the second of background before it is also
    the wake word.

    The lead is sized from `receptive`, the ~2.0 s of audio a scored
    window really covers, so that at least one window both begins before
    the phrase and ends just after it. Sizing it from the 1.28 s stride
    instead is what produced a training set with zero positives.
    """
    hop = cfg.hop_samples
    tail = rng.randint(0, 3) * hop          # how late the detector notices
    lead = max(hop, receptive + hop - len(phrase) - tail)
    total = lead + len(phrase) + tail + 3 * hop

    bg = np.zeros(total, dtype=np.float32)
    if background.size:
        reps = int(np.ceil(total / len(background)))
        bg = np.tile(background, reps)[:total].astype(np.float32)
        # SNR in dB against the phrase, drawn per example. 20 dB is a
        # quiet room; 0 dB is someone talking at the same volume in it.
        snr = rng.choice([30.0, 20.0, 15.0, 10.0, 5.0, 0.0])
        p_rms = float(np.sqrt(np.mean(phrase**2))) + 1e-9
        b_rms = float(np.sqrt(np.mean(bg**2))) + 1e-9
        bg *= (p_rms / b_rms) / (10 ** (snr / 20.0))

    clip = bg.copy()
    clip[lead:lead + len(phrase)] += phrase * rng.uniform(0.4, 1.0)
    peak = float(np.max(np.abs(clip)))
    if peak > 1.0:
        clip /= peak
    return clip.astype(np.float32), lead, lead + len(phrase)


def positive_windows(emb: Embedder, clip: np.ndarray, start: int, end: int,
                     cfg: WakeConfig) -> np.ndarray:
    """Only the windows a live detector could plausibly fire on.

    A window counts when it contains the whole phrase and ends soon after
    it does: it must begin at or before the phrase begins, end at or
    after the phrase ends, and end within four hops of that. Anything
    else is a window of mostly-background that happens to share a file
    with a positive, and labelling it positive is how a head learns to
    fire on the silence before the phrase.

    The spans come from `Embedder.window_span`, which uses the measured
    receptive field rather than the stride — see the comment on
    `Embedder.receptive_samples` for the run this cost.
    """
    hop, n = cfg.hop_samples, cfg.embedding_window
    all_windows = emb.windows(clip)
    keep = []
    for i in range(len(all_windows)):
        w_start, w_end = emb.window_span(i)
        if w_start <= start and end <= w_end <= end + 4 * hop:
            keep.append(all_windows[i])
    return np.asarray(keep, dtype=np.float32) if keep else np.zeros((0, n, 96), np.float32)


class Head(__import__("torch").nn.Module):
    """Flatten, two hidden layers, one logit — the same shape as the
    openWakeWord heads this front end was built for (their exported
    graphs start with a Flatten on a [1,16,96] input).
    """

    def __init__(self, n_in: int = 16 * 96, hidden: int = 128) -> None:
        from torch import nn

        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        import torch

        # Sigmoid is INSIDE the exported graph so `wake.py`'s threshold is
        # a probability in config, not a raw logit whose scale changes
        # every time this is retrained.
        return torch.sigmoid(self.net(x))


def false_accepts_per_hour(scores: np.ndarray, threshold: float, cfg: WakeConfig) -> float:
    """The number that decides whether this ships.

    Each scored window is one hop of audio — 80 ms — so the hours of
    speech behind a set of scores is just the count times the hop. A
    detector is judged on how often it interrupts you, not on accuracy.
    """
    hours = len(scores) * cfg.hop_samples / SAMPLE_RATE / 3600.0
    return float((scores >= threshold).sum()) / max(hours, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "models" / "wake-hey-elizabeth")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "prepared" / "wake")
    ap.add_argument("--librispeech", type=Path,
                    default=REPO / "data" / "datasets" / "librispeech-asr-corpus" / "clean")
    ap.add_argument("--shards", type=int, default=6, help="parquet shards to sample")
    ap.add_argument("--per-shard", type=int, default=60, help="clips per shard")
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.85, 1.0, 1.15])
    ap.add_argument("--repeats", type=int, default=6, help="augmentations per positive render")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-voices", type=int, default=0,
                    help="0 = every English Kokoro voice; a small number for a smoke run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Elizabeth().wake
    t0 = time.monotonic()

    voices = english_voices(REPO / "models")
    if args.max_voices:
        voices = voices[:args.max_voices]
    print(f"voices: {len(voices)} English of Kokoro's 54", flush=True)

    print("1/5 synthesising the phrase and its confusables", flush=True)
    pos_files = synthesise(POSITIVE_TEXTS, voices, args.speeds, args.cache / "pos")
    neg_files = synthesise(NEGATIVE_TEXTS, voices, [1.0], args.cache / "neg")
    print(f"  {len(pos_files)} positive renders, {len(neg_files)} confusables"
          f"  [{time.monotonic()-t0:.0f}s]", flush=True)

    print("2/5 reading LibriSpeech for background and hard negatives", flush=True)
    speech = list(librispeech_clips(args.librispeech, args.shards, args.per_shard, rng))
    split = int(len(speech) * 0.75)
    bg_pool, held_out = speech[:split], speech[split:]
    print(f"  {len(bg_pool)} background clips, {len(held_out)} held out for the metric"
          f"  [{time.monotonic()-t0:.0f}s]", flush=True)

    emb = Embedder(cfg)
    import soundfile as sf

    print("3/5 embedding", flush=True)
    X, y = [], []
    for i, f in enumerate(pos_files, 1):
        phrase, _ = sf.read(f, dtype="float32")
        for _ in range(args.repeats):
            bg = bg_pool[rng.randrange(len(bg_pool))] if rng.random() < 0.8 else np.zeros(0, np.float32)
            clip, s, e = augment(phrase, bg, rng, cfg, emb.receptive_samples)
            w = positive_windows(emb, clip, s, e, cfg)
            if len(w):
                X.append(w); y.append(np.ones(len(w), dtype=np.float32))
        if i % 50 == 0:
            print(f"  positives {i}/{len(pos_files)}  [{time.monotonic()-t0:.0f}s]", flush=True)

    for i, f in enumerate(neg_files, 1):
        phrase, _ = sf.read(f, dtype="float32")
        for _ in range(max(2, args.repeats // 2)):
            bg = bg_pool[rng.randrange(len(bg_pool))] if rng.random() < 0.8 else np.zeros(0, np.float32)
            clip, _s, _e = augment(phrase, bg, rng, cfg, emb.receptive_samples)
            w = emb.windows(clip)
            if len(w):
                X.append(w); y.append(np.zeros(len(w), dtype=np.float32))
        if i % 50 == 0:
            print(f"  confusables {i}/{len(neg_files)}  [{time.monotonic()-t0:.0f}s]", flush=True)

    for i, clip in enumerate(bg_pool, 1):
        w = emb.windows(clip)
        if len(w):
            X.append(w); y.append(np.zeros(len(w), dtype=np.float32))
        if i % 50 == 0:
            print(f"  bulk speech {i}/{len(bg_pool)}  [{time.monotonic()-t0:.0f}s]", flush=True)

    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)
    if y.sum() < 100:
        # The first run of this script produced ZERO positive windows and
        # trained happily on them, reporting recall as nan across the whole
        # sweep. A dataset with no positives is a bug every time.
        raise SystemExit(
            f"only {int(y.sum())} positive windows out of {len(y)} — the positive "
            "alignment is wrong, not the model. Check Embedder.receptive_samples."
        )
    print(f"  {len(X)} windows, {int(y.sum())} positive ({y.mean()*100:.1f}%)"
          f"  [{time.monotonic()-t0:.0f}s]", flush=True)

    idx = np.arange(len(X)); np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    cut = int(len(X) * 0.9)
    Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]

    print("4/5 fitting the head on CPU", flush=True)
    model = Head()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # The classes are wildly unbalanced by construction -- hours of speech
    # against a few thousand positive windows -- so the loss is weighted
    # rather than the data resampled, which would throw away negatives
    # that are the entire point of the false-accept metric.
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)])
    lossf = torch.nn.BCELoss(reduction="none")
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)[:, None]
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)[:, None]
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 512):
            sel = perm[b:b + 512]
            opt.zero_grad()
            out = model(Xtr_t[sel])
            w = torch.where(ytr_t[sel] > 0.5, pos_weight, torch.ones(1))
            (lossf(out, ytr_t[sel]) * w).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva_t)
            w = torch.where(yva_t > 0.5, pos_weight, torch.ones(1))
            vl = float((lossf(out, yva_t) * w).mean())
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} val {vl:.4f}{'  *' if vl == best else ''}", flush=True)
    model.load_state_dict(best_state)

    print("5/5 scoring on held-out speech and exporting", flush=True)
    model.eval()
    ho = [emb.windows(c) for c in held_out]
    ho = np.concatenate([w for w in ho if len(w)]).astype(np.float32)
    with torch.no_grad():
        neg_scores = model(torch.from_numpy(ho)).numpy().reshape(-1)
        pos_scores = model(torch.from_numpy(X[y > 0.5])).numpy().reshape(-1)

    sweep = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        sweep.append({
            "threshold": th,
            "recall": float((pos_scores >= th).mean()),
            "false_accepts_per_hour": false_accepts_per_hour(neg_scores, th, cfg),
        })
    hours = len(neg_scores) * cfg.hop_samples / SAMPLE_RATE / 3600.0
    print(f"  held-out speech: {hours:.2f} h, {len(neg_scores)} windows")
    print("  threshold   recall   false accepts/hour")
    for r in sweep:
        print(f"    {r['threshold']:.2f}      {r['recall']*100:5.1f}%   "
              f"{r['false_accepts_per_hour']:8.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, cfg.embedding_window, 96)
    torch.onnx.export(model, (dummy,), str(args.out / "head.onnx"),
                      input_names=["embeddings"], output_names=["score"], dynamo=False)
    (args.out / "report.json").write_text(json.dumps({
        "phrase": cfg.phrase,
        "windows": len(X),
        "positive_windows": int(y.sum()),
        "held_out_hours": hours,
        "sweep": sweep,
        "voices": voices,
        "seconds": round(time.monotonic() - t0, 1),
    }, indent=2))
    print(f"wrote {args.out/'head.onnx'} and report.json  "
          f"[{time.monotonic()-t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
