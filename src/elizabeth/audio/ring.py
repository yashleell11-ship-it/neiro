"""A fixed-capacity circular buffer of float32 mono audio.

First used by Task 4's terminal push-to-talk for pre-roll (capture runs
continuously; the last `pre_roll_s` seconds already in the ring get
prepended the instant you press the key, so the first syllable is never
clipped). Stage 1's real endpointer reuses this exact class at a larger
capacity — see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity_seconds: float, samplerate: int) -> None:
        self.samplerate = samplerate
        self._capacity = max(1, int(capacity_seconds * samplerate))
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = chunk.shape[0]
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                # the chunk alone overflows the whole buffer — keep only
                # its most recent tail
                self._buf[:] = chunk[-self._capacity :]
                self._write_pos = 0
                self._filled = self._capacity
                return
            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = chunk
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = chunk[:first]
                self._buf[: end - self._capacity] = chunk[first:]
            self._write_pos = end % self._capacity
            self._filled = min(self._capacity, self._filled + n)

    def read_last(self, seconds: float) -> np.ndarray:
        """The most recent `seconds` of audio, oldest sample first.
        Returns fewer samples than requested if the buffer hasn't been
        filled that long yet — never zero-pads, since that would splice
        fake silence in front of a real recording.
        """
        with self._lock:
            n = min(int(seconds * self.samplerate), self._filled)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start : start + n].copy()
            first = self._capacity - start
            return np.concatenate([self._buf[start:], self._buf[: n - first]])
