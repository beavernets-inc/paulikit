# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Tests for the binary chunk-framed checkpoint format
(docs/superpowers/specs/2026-09-08-binary-chunk-framed-checkpoint-design.md).

These test the format primitives directly - header encoding, frame
round-trip, and recovery from truncated or corrupt files - separately
from the decomposition behaviour that the four existing checkpoint
test files already pin.
"""

import struct

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _CHECKPOINT_HEADER_STRUCT,
    _CHECKPOINT_MAGIC,
    _CHECKPOINT_VERSION,
    _append_checkpoint_frame,
    _append_progress_record,
    _checkpoint_frame_header,
    _iter_checkpoint_frames,
    _parse_checkpoint_frame_header,
)


def test_header_is_24_bytes():
    assert _CHECKPOINT_HEADER_STRUCT.size == 24


@pytest.mark.parametrize(
    "dtype", [np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.intp)]
)
def test_header_round_trip_every_index_dtype(dtype):
    raw = _checkpoint_frame_header(chunk_index=7, n_terms=1234, idx_dtype=dtype)
    assert len(raw) == 24
    chunk_index, n_terms, got = _parse_checkpoint_frame_header(raw)
    assert (chunk_index, n_terms) == (7, 1234)
    assert got == dtype


def test_header_round_trip_large_values():
    # chunk_index and n_terms are u64; N=150 has ~91.6M terms overall
    # and thousands of chunks, so neither may be narrowed to u32.
    raw = _checkpoint_frame_header(
        chunk_index=2**40, n_terms=91_652_096, idx_dtype=np.dtype(np.uint32)
    )
    chunk_index, n_terms, _ = _parse_checkpoint_frame_header(raw)
    assert chunk_index == 2**40
    assert n_terms == 91_652_096


def test_parse_rejects_wrong_magic():
    raw = bytearray(
        _checkpoint_frame_header(0, 1, np.dtype(np.uint32))
    )
    raw[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        _parse_checkpoint_frame_header(bytes(raw))


def test_parse_rejects_unknown_version():
    raw = _CHECKPOINT_HEADER_STRUCT.pack(
        _CHECKPOINT_MAGIC, _CHECKPOINT_VERSION + 1, 1, 0, 0, 1
    )
    with pytest.raises(ValueError, match="version"):
        _parse_checkpoint_frame_header(raw)


def test_parse_rejects_unknown_index_dtype_code():
    # An unknown width must raise rather than default to a guess:
    # guessing wrong silently wraps indices and yields wrong labels.
    raw = _CHECKPOINT_HEADER_STRUCT.pack(
        _CHECKPOINT_MAGIC, _CHECKPOINT_VERSION, 99, 0, 0, 1
    )
    with pytest.raises(ValueError, match="index dtype"):
        _parse_checkpoint_frame_header(raw)


def test_header_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="index dtype"):
        _checkpoint_frame_header(0, 1, np.dtype(np.float64))


def test_parse_rejects_short_buffer():
    with pytest.raises(ValueError, match="truncated"):
        _parse_checkpoint_frame_header(b"PKCP")


def _write_frames(path, frames, idx_dtype=np.dtype(np.uint32)):
    for chunk_index, x, z, coeff in frames:
        _append_checkpoint_frame(
            path, chunk_index,
            np.asarray(x, dtype=idx_dtype),
            np.asarray(z, dtype=idx_dtype),
            np.asarray(coeff, dtype=complex),
            idx_dtype,
        )


def test_frame_round_trip_preserves_values_and_dtypes(tmp_path):
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [
        (0, [1, 2, 3], [4, 5, 6], [1 + 2j, 3 + 0j, -1j]),
        (1, [7], [8], [0.5 + 0j]),
    ])
    got = list(_iter_checkpoint_frames(path))
    assert [g[0] for g in got] == [0, 1]
    np.testing.assert_array_equal(got[0][1], [1, 2, 3])
    np.testing.assert_array_equal(got[0][2], [4, 5, 6])
    np.testing.assert_allclose(got[0][3], [1 + 2j, 3 + 0j, -1j])
    assert got[0][3].dtype == np.dtype(complex)
    np.testing.assert_array_equal(got[1][1], [7])


def test_frame_round_trip_empty_chunk(tmp_path):
    # A chunk where every coefficient fell below atol survives with
    # zero terms; it must still be recorded so its index is replayable.
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [(0, [], [], [])])
    got = list(_iter_checkpoint_frames(path))
    assert len(got) == 1
    assert got[0][0] == 0
    assert len(got[0][1]) == 0


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.intp])
def test_frame_round_trip_every_index_dtype(tmp_path, dtype):
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [(0, [1, 2], [3, 4], [1j, 2j])],
                  idx_dtype=np.dtype(dtype))
    got = list(_iter_checkpoint_frames(path))
    np.testing.assert_array_equal(got[0][1], [1, 2])
    assert got[0][1].dtype == np.dtype(dtype)


@pytest.mark.parametrize("cut", [1, 10, 24, 30, 40])
def test_truncated_tail_recovers_every_complete_frame(tmp_path, cut):
    # Truncation mid-header, exactly at a boundary, and mid-payload
    # must all recover frame 0 intact and simply drop the partial.
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [(0, [1, 2], [3, 4], [1j, 2j])])
    complete = path.read_bytes()
    _write_frames(path, [(1, [9], [9], [9j])])
    full = path.read_bytes()
    path.write_bytes(full[: len(complete) + cut])

    got = list(_iter_checkpoint_frames(path))
    assert got[0][0] == 0
    np.testing.assert_array_equal(got[0][1], [1, 2])
    assert all(g[0] == 0 for g in got), "partial frame must not be yielded"


def test_corruption_in_an_early_frame_is_detected(tmp_path):
    # JSONL could only detect a truncated FINAL line; frame magic makes
    # corruption anywhere detectable.
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [(0, [1], [1], [1j]), (1, [2], [2], [2j])])
    raw = bytearray(path.read_bytes())
    raw[0:4] = b"XXXX"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="magic"):
        list(_iter_checkpoint_frames(path))


def test_frames_absent_from_valid_indices_are_dropped(tmp_path):
    path = tmp_path / "ckpt.bin"
    _write_frames(path, [(0, [1], [1], [1j]), (1, [2], [2], [2j])])
    got = list(_iter_checkpoint_frames(path, valid_indices={0}))
    assert [g[0] for g in got] == [0]


def test_missing_file_yields_nothing(tmp_path):
    assert list(_iter_checkpoint_frames(tmp_path / "absent.bin")) == []


def test_sequential_resume_replays_per_chunk_not_one_combined_tile(tmp_path):
    # The JSONL format could not preserve chunk boundaries, so resume
    # yielded one combined tile. Frames record chunk_index per frame,
    # so replay is per original chunk - which is what keeps replay
    # memory bounded by the largest chunk rather than by the file.
    from paulikit.algorithms.fwht import _load_checkpoint

    path = tmp_path / "seq.bin"
    _write_frames(path, [
        (0, [1, 2], [3, 4], [1j, 2j]),
        (1, [5], [6], [3j]),
    ])
    progress_path = tmp_path / "seq.bin.progress.json"
    _append_progress_record(progress_path, 0)
    _append_progress_record(progress_path, 1)

    next_chunk, frames = _load_checkpoint(path)
    assert next_chunk == 2
    replayed = list(frames)
    assert len(replayed) == 2, "replay must be per chunk, not combined"
    np.testing.assert_array_equal(replayed[0][0], [1, 2])
    np.testing.assert_array_equal(replayed[1][0], [5])


def test_resume_memory_is_bounded_by_chunk_not_file(tmp_path):
    # The JSONL reader called f.readlines() and built three full Python
    # lists plus a dedup dict, so resume memory scaled with FILE size -
    # >20 GB at N=150 on a 15 GB machine. Frames must scale with the
    # largest CHUNK instead. Many small frames, one peak measurement.
    import tracemalloc

    path = tmp_path / "many.bin"
    n_frames, per_frame = 200, 500
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        _append_checkpoint_frame(
            path, i,
            rng.integers(0, 1000, per_frame).astype(np.uint32),
            rng.integers(0, 1000, per_frame).astype(np.uint32),
            rng.standard_normal(per_frame).astype(complex),
            np.dtype(np.uint32),
        )
    file_bytes = path.stat().st_size

    tracemalloc.start()
    total = 0
    for _index, x, _z, _coeff in _iter_checkpoint_frames(path):
        total += len(x)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert total == n_frames * per_frame, "every frame must be read"
    # Bounded means the peak tracks one frame, not the whole file. A
    # generous 25% bound still fails loudly for a readlines()-style
    # reader, which would peak at several times the file size.
    assert peak < file_bytes * 0.25, (
        f"resume peak {peak} should be well below file size {file_bytes}"
    )


@pytest.mark.parametrize("n_terms", [1, 3, 4])
def test_replayed_coefficients_are_aligned(tmp_path, n_terms):
    # With uint16 indices and an odd term count the coefficients start
    # at a payload offset that is not a multiple of 8. np.frombuffer
    # then yields an unaligned complex128 array, which the native
    # Hermiticity check reads as const double * - undefined behaviour
    # that UBSan reports and strict-alignment CPUs can fault on.
    from paulikit.algorithms.fwht import (
        _append_checkpoint_frame,
        _iter_checkpoint_frames,
    )

    path = tmp_path / "ckpt.bin"
    x = np.arange(n_terms, dtype=np.uint16)
    z = np.arange(n_terms, dtype=np.uint16)[::-1].copy()
    coeff = (np.arange(n_terms) + 0.25j * np.arange(n_terms)).astype(complex)
    _append_checkpoint_frame(path, 0, x, z, coeff, np.dtype(np.uint16))
    ((index, rx, rz, rc),) = list(_iter_checkpoint_frames(path))
    assert index == 0
    assert rc.flags.aligned
    np.testing.assert_array_equal(rx, x)
    np.testing.assert_array_equal(rz, z)
    np.testing.assert_array_equal(rc, coeff)
