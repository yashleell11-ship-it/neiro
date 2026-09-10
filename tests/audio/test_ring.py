"""Regression tests for RingBuffer (src/neiro/audio/ring.py).

This is the piece Task 4's pre-roll depends on being exactly right —
get the wraparound math wrong and the failure mode is subtle (the first
syllable gets clipped some of the time, or the wrong samples get spliced
in), not a crash. Worth pinning down with real assertions before Stage 1
builds the real endpointer on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from neiro.audio.ring import RingBuffer


def test_empty_buffer_reads_nothing() -> None:
    ring = RingBuffer(capacity_seconds=1.0, samplerate=100)
    assert ring.read_last(0.5).shape == (0,)


def test_read_more_than_written_returns_only_whats_filled() -> None:
    # Never zero-pad — that would splice fake silence in front of real
    # audio, which is worse than just returning less pre-roll than asked.
    ring = RingBuffer(capacity_seconds=1.0, samplerate=100)
    ring.write(np.ones(20, dtype=np.float32))
    got = ring.read_last(1.0)  # asking for 100 samples' worth
    assert got.shape == (20,)
    assert np.all(got == 1.0)


def test_read_last_returns_most_recent_samples_in_order() -> None:
    ring = RingBuffer(capacity_seconds=1.0, samplerate=100)
    ring.write(np.arange(100, dtype=np.float32))
    got = ring.read_last(0.3)  # last 30 samples
    np.testing.assert_array_equal(got, np.arange(70, 100, dtype=np.float32))


def test_wraparound_stitches_correctly() -> None:
    # capacity 10, write 7 then 7 more -> the second write wraps past
    # the end of the underlying array. read_last must stitch the tail
    # and the head back together in the right order.
    ring = RingBuffer(capacity_seconds=1.0, samplerate=10)  # capacity = 10
    ring.write(np.arange(7, dtype=np.float32))  # 0..6, fills to 7
    ring.write(np.arange(100, 107, dtype=np.float32))  # 100..106, wraps

    got = ring.read_last(
        1.0
    )  # ask for everything (10 samples, but only wrote 14 total -> capacity 10)
    # buffer holds the most recent 10 samples of the 14 written:
    # [0,1,2,3,4,5,6,100,101,102,103,104,105,106] -> last 10 ->
    # [4,5,6,100,101,102,103,104,105,106]
    expected = np.array([4, 5, 6, 100, 101, 102, 103, 104, 105, 106], dtype=np.float32)
    np.testing.assert_array_equal(got, expected)


def test_single_chunk_larger_than_capacity_keeps_only_the_tail() -> None:
    ring = RingBuffer(capacity_seconds=1.0, samplerate=5)  # capacity = 5
    ring.write(np.arange(20, dtype=np.float32))  # one big chunk, way over capacity
    got = ring.read_last(1.0)
    np.testing.assert_array_equal(got, np.arange(15, 20, dtype=np.float32))


def test_many_small_writes_behave_like_one_big_write() -> None:
    ring_many = RingBuffer(capacity_seconds=1.0, samplerate=50)
    ring_one = RingBuffer(capacity_seconds=1.0, samplerate=50)
    data = np.arange(123, dtype=np.float32)

    for i in range(0, 123, 7):  # odd chunk size on purpose
        ring_many.write(data[i : i + 7])
    ring_one.write(data)

    np.testing.assert_array_equal(ring_many.read_last(1.0), ring_one.read_last(1.0))


def test_partial_read_after_wraparound() -> None:
    ring = RingBuffer(capacity_seconds=1.0, samplerate=10)  # capacity 10
    ring.write(np.arange(15, dtype=np.float32))  # wraps once; holds last 10: 5..14
    got = ring.read_last(0.3)  # last 3 samples: 12, 13, 14
    np.testing.assert_array_equal(got, np.array([12, 13, 14], dtype=np.float32))


def test_write_accepts_2d_column_and_flattens() -> None:
    # sounddevice callbacks hand back shape (frames, channels); the
    # caller is expected to slice to mono before calling write(), but
    # RingBuffer shouldn't silently misbehave on an accidental (N, 1).
    ring = RingBuffer(capacity_seconds=1.0, samplerate=10)
    ring.write(np.ones((5, 1), dtype=np.float32))
    assert ring.read_last(1.0).shape == (5,)


@pytest.mark.parametrize("capacity_seconds,samplerate", [(0.001, 8000), (2.0, 1)])
def test_capacity_is_never_zero(capacity_seconds: float, samplerate: int) -> None:
    # a degenerate config (very short capacity, or a silly samplerate)
    # must still produce a usable buffer, not a zero-size array that
    # every write() then overflows into a no-op.
    ring = RingBuffer(capacity_seconds=capacity_seconds, samplerate=samplerate)
    ring.write(np.ones(3, dtype=np.float32))
    assert ring.read_last(10.0).size >= 1
