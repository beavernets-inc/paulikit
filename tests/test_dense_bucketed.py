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
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
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
