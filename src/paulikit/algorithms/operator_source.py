# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Operator input sources for chunked Pauli FWHT gathers.

Hot-path module: only the adapters ``fwht`` needs at import time
(``DenseResidentSource``, ``SparseScatterSource``). Spill / file-backed
dense lives in ``dense_bucketed`` and must be imported only when that
path is opted into — same discipline as the lazy ``scipy.sparse``
import in ``fwht.py``.

Contract::

    gathered[row, q] = H[(chunk_start + row) ⊕ q, q]
"""

from __future__ import annotations

import numpy as np

try:
    from paulikit._native import gather_native as _gather_native
except ImportError:  # pragma: no cover - extension optional at import
    _gather_native = None


class DenseResidentSource:
    """XOR-gather from a fully resident C-contiguous dense operator."""

    __slots__ = ("_operator", "_dim", "_z_range")

    def __init__(self, operator):
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(
                f"operator must be square, got shape {operator.shape}"
            )
        self._operator = np.ascontiguousarray(operator, dtype=complex)
        self._dim = int(self._operator.shape[0])
        self._z_range = np.arange(self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def operator(self):
        return self._operator

    def gather_chunk(self, chunk_start: int, n_rows: int):
        if n_rows < 0 or chunk_start < 0 or chunk_start + n_rows > self._dim:
            raise ValueError(
                f"chunk [{chunk_start}, {chunk_start + n_rows}) out of "
                f"range for dim={self._dim}"
            )
        if n_rows == 0:
            return np.empty((0, self._dim), dtype=complex)
        if _gather_native is not None:
            return _gather_native.gather_dense_chunk(
                self._operator, chunk_start, n_rows
            )
        x_values = np.arange(chunk_start, chunk_start + n_rows)
        q_range = self._z_range
        p_indices = x_values[:, np.newaxis] ^ q_range[np.newaxis, :]
        return np.ascontiguousarray(
            self._operator[p_indices, q_range[np.newaxis, :]]
        )


class SparseScatterSource:
    """Scatter pre-extracted nnz values into a zeroed gather tile."""

    __slots__ = (
        "_dim",
        "_sorted_inverse",
        "_sorted_q_nz",
        "_sorted_values",
    )

    def __init__(
        self,
        dim: int,
        sorted_inverse,
        sorted_q_nz,
        sorted_values,
    ):
        self._dim = int(dim)
        self._sorted_inverse = sorted_inverse
        self._sorted_q_nz = sorted_q_nz
        self._sorted_values = sorted_values

    @property
    def dim(self) -> int:
        return self._dim

    def gather_chunk(self, chunk_start: int, n_rows: int):
        chunk_end = chunk_start + n_rows
        lo = int(np.searchsorted(self._sorted_inverse, chunk_start))
        hi = int(np.searchsorted(self._sorted_inverse, chunk_end))
        gathered_chunk = np.zeros((n_rows, self._dim), dtype=complex)
        if hi > lo:
            gathered_chunk[
                self._sorted_inverse[lo:hi] - chunk_start,
                self._sorted_q_nz[lo:hi],
            ] = self._sorted_values[lo:hi]
        return gathered_chunk
