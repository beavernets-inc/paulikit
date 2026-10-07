# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Tests for OperatorSource gather adapters."""

import numpy as np
import pytest

from paulikit.algorithms.operator_source import (
    DenseResidentSource,
    SparseScatterSource,
)


def _xor_gather_reference(operator: np.ndarray, chunk_start: int, n_rows: int):
    dim = operator.shape[0]
    out = np.empty((n_rows, dim), dtype=complex)
    for r in range(n_rows):
        x = chunk_start + r
        for q in range(dim):
            out[r, q] = operator[x ^ q, q]
    return out


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_dense_resident_matches_xor_definition(n_qubits):
    rng = np.random.default_rng(0)
    dim = 2**n_qubits
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    operator = (raw + raw.conj().T) / 2
    source = DenseResidentSource(operator)
    assert source.dim == dim
    for chunk_start, n_rows in [(0, 1), (0, dim), (dim // 2, max(1, dim // 4))]:
        got = source.gather_chunk(chunk_start, n_rows)
        expected = _xor_gather_reference(operator, chunk_start, n_rows)
        np.testing.assert_array_equal(got, expected)
        assert got.flags.c_contiguous


def test_dense_resident_rejects_bad_chunk():
    source = DenseResidentSource(np.eye(4, dtype=complex))
    with pytest.raises(ValueError):
        source.gather_chunk(3, 2)


def test_sparse_scatter_matches_manual_placement():
    dim = 4
    # Nonzeros belonging to x=1 (p^q=1) and x=2.
    # For (p,q)=(0,1): x=1; (p,q)=(3,1): x=2; (p,q)=(2,0): x=2.
    # inverse index into active_x=arange is just x itself when fully dense
    # active set; here we treat sorted_inverse as the x values of nnz.
    sorted_inverse = np.array([1, 2, 2], dtype=np.intp)
    sorted_q_nz = np.array([1, 1, 0], dtype=np.intp)
    sorted_values = np.array([10 + 0j, 20 + 0j, 30 + 0j])
    source = SparseScatterSource(dim, sorted_inverse, sorted_q_nz, sorted_values)

    tile = source.gather_chunk(1, 2)  # rows for x=1 and x=2
    assert tile.shape == (2, dim)
    assert tile[0, 1] == 10 + 0j
    assert tile[1, 1] == 20 + 0j
    assert tile[1, 0] == 30 + 0j
    assert tile[0, 0] == 0
