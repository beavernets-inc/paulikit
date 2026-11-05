# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Numeric construction of the coupled-oscillator Hamiltonian.

Built directly in floating point, so large N carries no symbolic
overhead. The N=2 and N=4 cases are pinned by the fixtures in
``paulikit.testing.fixtures``, which the test suite checks against a
decomposition and reconstruction round trip.

This is the operator paulikit's own fixtures and benchmarks decompose.
It is a convenience for exercising the package, not a required input:
the decomposition entry points accept any square complex matrix whose
dimension is a power of two, Hermitian or otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# scipy.sparse is imported LAZILY, not at module load time. Measured
# directly (perf stat instructions:u): `import scipy.sparse` alone
# costs several hundred million instructions on top of bare NumPy -
# paid previously by every caller of this module, including every
# caller who only ever passes sparse=False (the default) and never
# touches scipy at all. Both functions below already take `sparse` as
# an explicit, caller-controlled parameter defaulting to False, so
# there is no need for any duck-typing to decide whether scipy is
# needed - the parameter already says so.
_sp = None
_sp_import_attempted = False


def _scipy_sparse_module():
    """Returns the ``scipy.sparse`` module, importing it on first call
    and caching the result (including the "not installed" case, so a
    missing scipy is retried at most once, not on every call)."""
    global _sp, _sp_import_attempted
    if not _sp_import_attempted:
        _sp_import_attempted = True
        try:
            import scipy.sparse as sp_module
            _sp = sp_module
        except ImportError:
            _sp = None
    return _sp


def build_hamiltonian(
    n_oscillators: int,
    spring_constants: dict[tuple[int, int], float],
    masses: list[float],
    sparse: bool = False,
) -> NDArray[np.floating]:
    """Build the coupled-oscillator Hamiltonian matrix.

    An ``(N + N*(N+1)/2)``-dimensional real symmetric matrix
    ``H = -[[0, B], [B^T, 0]]``, following Babbush, Berry, Kothari,
    Somma, Wiebe, Phys. Rev. X 13, 041041 (2023), arXiv:2303.13012,
    Sec. III. ``B`` is ``N x N*(N+1)/2`` with one column per spring:
    the N wall springs ``(i, i)`` first, then every coupling ``(j, k)``
    with ``j < k`` in the order ``for j in range(N), for k in
    range(j+1, N)``. Column ``(i, i)`` holds ``sqrt(k_ii/m_i)`` in row
    i; column ``(j, k)`` holds ``sqrt(k_jk/m_j)`` in row j and
    ``-sqrt(k_jk/m_k)`` in row k. Then ``B B^T = M^{-1/2} F M^{-1/2}``,
    with F the stiffness matrix of Newton's equations, so evolving
    under H reproduces the oscillators' dynamics.

    Args:
        n_oscillators: Number of coupled oscillators, N.
        spring_constants: Mapping from ``(i, j)`` with ``i <= j`` to
            the spring constant k_ij. Diagonal entries ``(i, i)`` are
            the on-site spring constants; off-diagonal entries are
            the coupling terms between oscillator i and j. Every pair
            with ``i <= j`` must be present - a missing one raises
            ``KeyError`` rather than defaulting to zero, so that an
            incompletely specified system fails loudly.
        masses: Sequence of N masses, ``masses[i]`` for oscillator i.
        sparse: If ``False`` (default), returns a dense
            ``numpy.ndarray`` - unchanged behavior. If ``True``,
            returns a ``scipy.sparse.coo_matrix`` instead, built by
            assembling the same nonzero entries directly rather than
            scattering into a dense array first - avoids ever
            allocating the ``O(N**2)`` dense matrix. At N=150 the
            dense Hamiltonian alone costs ~4 GiB once padded and
            upcast to complex, despite being only 0.034% dense.
            Requires the ``scipy`` package (install via the
            ``paulikit[sparse]`` extra); raises ``ImportError`` if
            ``scipy`` is not installed - deliberately not a silent
            fallback to dense, since that would defeat the purpose of
            an explicit sparse request.

    Returns:
        A real-valued, symmetric matrix of shape
        ``(N + N*(N+1)//2, N + N*(N+1)//2)`` - a ``numpy.ndarray`` if
        ``sparse=False``, a ``scipy.sparse.coo_matrix`` if
        ``sparse=True``.

    Note:
        Before 0.1.4 every coupling column was placed in rows 0 and 1
        regardless of which oscillators the spring joins. That agrees
        with the construction above only for N=2, where (0, 1) is the
        only pair; for N >= 3 it produced the wrong dynamics.
    """
    sp_module = _scipy_sparse_module() if sparse else None
    if sparse and sp_module is None:
        raise ImportError(
            "sparse=True requires scipy, which is not installed. "
            "Install it via `pip install paulikit[sparse]`."
        )

    n = n_oscillators
    size = n + n * (n + 1) // 2

    # (row, col, value) of the upper-right block of H = -[[0,B],[B^T,0]];
    # the lower-left block is its transpose.
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i in range(n):
        rows.append(i)
        cols.append(n + i)
        values.append(-np.sqrt(spring_constants[(i, i)] / masses[i]))
    col = 2 * n
    for j in range(n):
        for k in range(j + 1, n):
            kappa = spring_constants[(j, k)]
            rows += [j, k]
            cols += [col, col]
            values += [-np.sqrt(kappa / masses[j]), np.sqrt(kappa / masses[k])]
            col += 1

    if not sparse:
        hamiltonian = np.zeros((size, size))
        hamiltonian[rows, cols] = values
        hamiltonian[cols, rows] = values
        return hamiltonian

    return sp_module.coo_matrix(
        (values + values, (rows + cols, cols + rows)), shape=(size, size)
    )


def pad_to_power_of_two(
    matrix: NDArray[np.floating],
    sparse: bool = False,
) -> tuple[NDArray[np.floating], int]:
    """Zero-pad a square matrix so its dimension is a power of two.

    Required before Pauli decomposition, since an n-qubit operator
    must have dimension exactly 2**n.

    Args:
        matrix: A square matrix - a ``numpy.ndarray`` if
            ``sparse=False``, or any ``scipy.sparse`` matrix (e.g. the
            ``coo_matrix`` returned by ``build_hamiltonian(...,
            sparse=True)``) if ``sparse=True``.
        sparse: If ``False`` (default), ``matrix`` and the return
            value are dense ``numpy.ndarray``s - unchanged behavior.
            If ``True``, ``matrix`` is treated as a ``scipy.sparse``
            matrix and the padded result is built via COO
            reconstruction at the new shape rather than
            ``csr_matrix.resize()`` - the latter mutates its input in
            place (verified directly), which would break this
            function's existing non-mutating contract. Requires the
            ``scipy`` package (see ``build_hamiltonian``'s
            ``sparse`` parameter for the same requirement/rationale).

    Returns:
        A tuple ``(padded_matrix, n_qubits)`` where ``padded_matrix``
        has shape ``(2**n_qubits, 2**n_qubits)`` and contains
        ``matrix`` in its top-left block, zero elsewhere -
        a ``numpy.ndarray`` if ``sparse=False``, a
        ``scipy.sparse.coo_matrix`` if ``sparse=True``.
    """
    sp_module = _scipy_sparse_module() if sparse else None
    if sparse and sp_module is None:
        raise ImportError(
            "sparse=True requires scipy, which is not installed. "
            "Install it via `pip install paulikit[sparse]`."
        )

    size = matrix.shape[0]
    n_qubits = int(np.ceil(np.log2(size)))
    dim = 2**n_qubits

    if not sparse:
        padded = np.zeros((dim, dim))
        padded[:size, :size] = matrix
        return padded, n_qubits

    coo = matrix.tocoo()
    padded = sp_module.coo_matrix((coo.data, (coo.row, coo.col)), shape=(dim, dim))
    return padded, n_qubits
