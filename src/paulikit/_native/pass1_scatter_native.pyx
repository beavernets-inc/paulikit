# cython: language_level=3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Cython binding for the Pass-1 bit-indexed spill scatter kernel.

Replaces the per-row NumPy mask scatter used while building spilled
dense buckets. See ``pass1_scatter.h`` for the bit-routing argument
(R = 2^k, q = p ⊕ (x_lo+r)).

Optional: ``dense_bucketed`` falls back to a pure-Python / NumPy bit
route if this module is absent, with identical results.
"""

import os

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t
from libc.errno cimport errno

cnp.import_array()

cdef extern from "pass1_scatter.h":
    void paulikit_pass1_scatter_block_into_bucket(
        const double *block_data, int64_t dim, int64_t p_start,
        int64_t n_p, int64_t x_lo, int64_t height, double *bucket_data
    ) nogil
    void paulikit_pass1_fill_bucket_from_operator(
        const double *operator_data, int64_t dim, int64_t x_lo,
        int64_t height, double *bucket_data
    ) nogil
    int paulikit_spill_write_c128(
        const char *path, const double *data, int64_t n_complex
    ) nogil


def scatter_block_into_bucket(
    cnp.ndarray block,
    *,
    int64_t p_start,
    int64_t x_lo,
    cnp.ndarray bucket,
):
    """Scatter a contiguous block of operator rows into one spill tile.

    Args:
        block: C-contiguous ``(n_p, dim)`` ``complex128`` — rows
            ``H[p_start]`` .. ``H[p_start + n_p - 1]``. Read only.
        p_start: First operator row index represented in ``block``.
        x_lo: First gather-row ``x`` covered by ``bucket``.
        bucket: C-contiguous ``(height, dim)`` ``complex128`` tile
            (caller-owned; written in place).

    Raises:
        ValueError: On bad dtype/layout/shape or out-of-range indices.
    """
    if block.ndim != 2:
        raise ValueError(f"block must be 2-D, got {block.ndim}-D")
    if bucket.ndim != 2:
        raise ValueError(f"bucket must be 2-D, got {bucket.ndim}-D")
    if block.dtype != np.complex128:
        raise ValueError(f"block must be complex128, got {block.dtype}")
    if bucket.dtype != np.complex128:
        raise ValueError(f"bucket must be complex128, got {bucket.dtype}")
    if not block.flags["C_CONTIGUOUS"]:
        raise ValueError("block must be C-contiguous")
    if not bucket.flags["C_CONTIGUOUS"]:
        raise ValueError("bucket must be C-contiguous")

    cdef int64_t n_p = block.shape[0]
    cdef int64_t dim = block.shape[1]
    cdef int64_t height = bucket.shape[0]
    cdef int64_t bucket_dim = bucket.shape[1]

    if bucket_dim != dim:
        raise ValueError(
            f"bucket width {bucket_dim} must match block dim {dim}"
        )
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of two, got {dim}")
    if p_start < 0 or n_p < 0 or p_start + n_p > dim:
        raise ValueError(
            f"block rows [{p_start}, {p_start + n_p}) out of range "
            f"for dim={dim}"
        )
    if x_lo < 0 or height < 0 or x_lo + height > dim:
        raise ValueError(
            f"tile x-range [{x_lo}, {x_lo + height}) out of range "
            f"for dim={dim}"
        )

    if n_p == 0 or height == 0:
        return

    cdef const double *block_data = <const double *> cnp.PyArray_DATA(block)
    cdef double *bucket_data = <double *> cnp.PyArray_DATA(bucket)

    with nogil:
        paulikit_pass1_scatter_block_into_bucket(
            block_data, dim, p_start, n_p, x_lo, height, bucket_data
        )


def fill_bucket_from_operator(cnp.ndarray operator, *, int64_t x_lo, cnp.ndarray bucket):
    """Fill one spill tile from a full dense ``(dim, dim)`` operator.

    ``operator`` may be a memmap (read-only C-contiguous complex128).
    Across all tiles covering ``[0, dim)``, each operator cell is read
    exactly once.
    """
    if operator.ndim != 2:
        raise ValueError(f"operator must be 2-D, got {operator.ndim}-D")
    if bucket.ndim != 2:
        raise ValueError(f"bucket must be 2-D, got {bucket.ndim}-D")
    if operator.dtype != np.complex128:
        raise ValueError(f"operator must be complex128, got {operator.dtype}")
    if bucket.dtype != np.complex128:
        raise ValueError(f"bucket must be complex128, got {bucket.dtype}")
    if not operator.flags["C_CONTIGUOUS"]:
        raise ValueError("operator must be C-contiguous")
    if not bucket.flags["C_CONTIGUOUS"]:
        raise ValueError("bucket must be C-contiguous")

    cdef int64_t dim = operator.shape[0]
    cdef int64_t dim1 = operator.shape[1]
    cdef int64_t height = bucket.shape[0]
    cdef int64_t bucket_dim = bucket.shape[1]

    if dim1 != dim:
        raise ValueError(
            f"operator must be square, got shape ({dim}, {dim1})"
        )
    if bucket_dim != dim:
        raise ValueError(
            f"bucket width {bucket_dim} must match operator dim {dim}"
        )
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"dim must be a power of two, got {dim}")
    if x_lo < 0 or height < 0 or x_lo + height > dim:
        raise ValueError(
            f"tile x-range [{x_lo}, {x_lo + height}) out of range "
            f"for dim={dim}"
        )

    if height == 0:
        return

    cdef const double *op_data = <const double *> cnp.PyArray_DATA(operator)
    cdef double *bucket_data = <double *> cnp.PyArray_DATA(bucket)

    with nogil:
        paulikit_pass1_fill_bucket_from_operator(
            op_data, dim, x_lo, height, bucket_data
        )


def write_c128_file(path, cnp.ndarray bucket):
    """Write a C-contiguous complex128 array with large POSIX writes.

    Byte-identical to ``bucket.tofile(path)`` for C-contiguous input.
    Releases the GIL for the duration of the write.
    """
    if bucket.dtype != np.complex128:
        raise ValueError(f"bucket must be complex128, got {bucket.dtype}")
    if not bucket.flags["C_CONTIGUOUS"]:
        raise ValueError("bucket must be C-contiguous")

    cdef int64_t n_complex = bucket.size
    cdef bytes path_b = os.fsencode(os.fspath(path))
    cdef const char *c_path = path_b
    cdef const double *data = <const double *> cnp.PyArray_DATA(bucket)
    cdef int rc

    with nogil:
        rc = paulikit_spill_write_c128(c_path, data, n_complex)
    if rc != 0:
        raise OSError(errno, os.strerror(errno), os.fsdecode(path_b))
