# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Bit-identical dense gather: resident XOR vs bucketed / spilled scatter."""

from pathlib import Path

import numpy as np
import pytest

from paulikit.algorithms.dense_bucketed import DenseBucketedSource
from paulikit.algorithms.operator_source import DenseResidentSource



def _random_hermitian(n_qubits: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (raw + raw.conj().T) / 2


def _assert_matches_resident(operator: np.ndarray, bucketed: DenseBucketedSource):
    dim = operator.shape[0]
    resident = DenseResidentSource(operator)
    assert bucketed.dim == dim
    for chunk_start in range(0, dim, bucketed.chunk_size):
        n_rows = min(bucketed.chunk_size, dim - chunk_start)
        a = resident.gather_chunk(chunk_start, n_rows)
        b = bucketed.gather_chunk(chunk_start, n_rows)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
def test_bucketed_matches_resident_gather(n_qubits, chunk_size):
    operator = _random_hermitian(n_qubits, seed=n_qubits * 17 + chunk_size)
    bucketed = DenseBucketedSource.from_array(operator, chunk_size=chunk_size)
    _assert_matches_resident(operator, bucketed)


def test_bucketed_refuses_misaligned_chunk_start():
    source = DenseBucketedSource.from_array(np.eye(4, dtype=complex), chunk_size=2)
    with pytest.raises(ValueError, match="multiple"):
        source.gather_chunk(1, 2)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_spill_path_bit_identical(tmp_path: Path, n_qubits, chunk_size):
    operator = _random_hermitian(n_qubits, seed=100 + n_qubits + chunk_size)
    bucketed = DenseBucketedSource.from_array(
        operator,
        chunk_size=chunk_size,
        spill_dir=tmp_path / "buckets",
        max_resident_buckets=1,
    )
    assert bucketed.spilled
    _assert_matches_resident(operator, bucketed)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_from_complex128_file_bit_identical(tmp_path: Path, n_qubits, chunk_size):
    operator = _random_hermitian(n_qubits, seed=200 + n_qubits * 3 + chunk_size)
    dim = operator.shape[0]
    path = tmp_path / "H.c128"
    # Row-major complex128 blob (same layout as ndarray.tofile default).
    np.ascontiguousarray(operator, dtype=np.complex128).tofile(path)

    bucketed = DenseBucketedSource.from_complex128_file(
        path,
        dim=dim,
        chunk_size=chunk_size,
        spill_dir=tmp_path / "file_buckets",
        max_resident_buckets=1,
    )
    assert bucketed.spilled
    _assert_matches_resident(operator, bucketed)


@pytest.mark.parametrize("n_qubits,chunk_size,spill_bucket_rows", [
    (4, 2, 4),
    (4, 2, 8),
    (5, 2, 8),
    (5, 4, 8),
])
def test_spill_bucket_rows_decoupled_drain_bit_identical(
    tmp_path: Path, n_qubits, chunk_size, spill_bucket_rows
):
    """Tall spill tiles + thin drain chunks must still match resident gather."""
    operator = _random_hermitian(
        n_qubits, seed=300 + n_qubits + chunk_size + spill_bucket_rows
    )
    bucketed = DenseBucketedSource.from_array(
        operator,
        chunk_size=chunk_size,
        spill_dir=tmp_path / "decoupled",
        spill_bucket_rows=spill_bucket_rows,
        max_resident_buckets=1,
    )
    assert bucketed.spilled
    assert bucketed.chunk_size == chunk_size
    assert bucketed.spill_bucket_rows == spill_bucket_rows
    n_spill = (operator.shape[0] + spill_bucket_rows - 1) // spill_bucket_rows
    assert len(list((tmp_path / "decoupled").glob("bucket_*.c128"))) == n_spill
    _assert_matches_resident(operator, bucketed)


def test_spill_bucket_rows_must_be_multiple_of_chunk_size(tmp_path: Path):
    operator = _random_hermitian(4, seed=7)
    with pytest.raises(ValueError, match="multiple"):
        DenseBucketedSource.from_array(
            operator,
            chunk_size=4,
            spill_dir=tmp_path / "bad",
            spill_bucket_rows=2,
        )


def test_spill_bucket_rows_must_be_power_of_two(tmp_path: Path):
    """Pass-1 bit routing requires R = 2^k (Hacker's Delight power-of-two test)."""
    operator = _random_hermitian(4, seed=8)
    with pytest.raises(ValueError, match="power of two"):
        DenseBucketedSource.from_array(
            operator,
            chunk_size=2,
            spill_dir=tmp_path / "bad_pow2",
            spill_bucket_rows=6,
        )


def test_bit_scatter_row_matches_mask_reference():
    """Bit-indexed q = p ⊕ (x_lo+r) must match the boolean-mask scatter."""
    from paulikit.algorithms.dense_bucketed import (
        _scatter_row_into_bucket,
        _scatter_row_into_bucket_bit,
    )

    dim = 32
    rng = np.random.default_rng(9)
    row = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    q_range = np.arange(dim)
    p = 13
    x_lo, height = 8, 8
    mask_bucket = np.zeros((height, dim), dtype=np.complex128)
    bit_bucket = np.zeros((height, dim), dtype=np.complex128)
    _scatter_row_into_bucket(
        mask_bucket, p=p, row=row, q_range=q_range, x_lo=x_lo, x_hi=x_lo + height
    )
    _scatter_row_into_bucket_bit(
        bit_bucket, p=p, row=row, x_lo=x_lo, height=height
    )
    np.testing.assert_array_equal(mask_bucket, bit_bucket)


def test_pass1_native_scatter_block_matches_bit_reference():
    """Compiled Pass-1 scatter must be bit-identical to the Python bit route."""
    scatter_native = pytest.importorskip(
        "paulikit._native.pass1_scatter_native",
        reason="pass1_scatter_native not built",
    )
    from paulikit.algorithms.dense_bucketed import _scatter_row_into_bucket_bit

    dim = 64
    height = 16
    x_lo = 32
    rng = np.random.default_rng(21)
    n_p = 8
    p_start = 5
    block = rng.standard_normal((n_p, dim)) + 1j * rng.standard_normal((n_p, dim))
    block = np.ascontiguousarray(block, dtype=np.complex128)

    expected = np.zeros((height, dim), dtype=np.complex128)
    for i in range(n_p):
        _scatter_row_into_bucket_bit(
            expected, p=p_start + i, row=block[i], x_lo=x_lo, height=height
        )

    got = np.zeros((height, dim), dtype=np.complex128)
    scatter_native.scatter_block_into_bucket(
        block, p_start=p_start, x_lo=x_lo, bucket=got
    )
    np.testing.assert_array_equal(expected, got)


def test_pass1_native_fill_bucket_matches_bit_reference():
    """One-touch fill_bucket_from_operator matches per-row bit scatter."""
    scatter_native = pytest.importorskip(
        "paulikit._native.pass1_scatter_native",
        reason="pass1_scatter_native not built",
    )
    from paulikit.algorithms.dense_bucketed import _scatter_row_into_bucket_bit

    dim = 32
    height = 8
    x_lo = 16
    rng = np.random.default_rng(22)
    H = np.ascontiguousarray(
        rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim)),
        dtype=np.complex128,
    )
    expected = np.zeros((height, dim), dtype=np.complex128)
    for p in range(dim):
        _scatter_row_into_bucket_bit(
            expected, p=p, row=H[p], x_lo=x_lo, height=height
        )
    got = np.zeros((height, dim), dtype=np.complex128)
    scatter_native.fill_bucket_from_operator(H, x_lo=x_lo, bucket=got)
    np.testing.assert_array_equal(expected, got)


def test_pass1_native_spill_write_matches_tofile(tmp_path: Path):
    """C spill write must be byte-identical to ndarray.tofile."""
    scatter_native = pytest.importorskip(
        "paulikit._native.pass1_scatter_native",
        reason="pass1_scatter_native not built",
    )
    rng = np.random.default_rng(23)
    bucket = np.ascontiguousarray(
        rng.standard_normal((64, 128)) + 1j * rng.standard_normal((64, 128)),
        dtype=np.complex128,
    )
    ref = tmp_path / "ref.c128"
    got = tmp_path / "got.c128"
    bucket.tofile(ref)
    scatter_native.write_c128_file(got, bucket)
    assert got.read_bytes() == ref.read_bytes()


def test_from_complex128_file_decoupled_bit_identical(tmp_path: Path):
    operator = _random_hermitian(5, seed=42)
    dim = operator.shape[0]
    path = tmp_path / "H.c128"
    np.ascontiguousarray(operator, dtype=np.complex128).tofile(path)
    bucketed = DenseBucketedSource.from_complex128_file(
        path,
        dim=dim,
        chunk_size=2,
        spill_bucket_rows=8,
        spill_dir=tmp_path / "file_decoupled",
        max_resident_buckets=1,
    )
    _assert_matches_resident(operator, bucketed)
