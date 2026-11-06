# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Fast known-answer tests: build an operator FROM chosen Pauli
coefficients, decompose it, and demand exactly those coefficients back.

The expected values are the inputs, written down by hand, so nothing
here is derived from paulikit itself or from an external library. The
tests take milliseconds, so they run on every CI job, native or
pure-Python fallback.

Label convention: character 0 of a label is the most significant
qubit, so the matrix of a label is ``np.kron`` of its letters in
reading order.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    fwht_pauli_terms,
    parallel_decompose_arrays,
    terms_from_arrays,
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# Mixed real and complex coefficients; includes the identity, single
# letters, strings with Y, and complex values on non-Hermitian terms
# (a complex coefficient on a Hermitian string is a non-Hermitian
# operator).
_KNOWN_6Q = {
    "IIIIII": 0.5,
    "XIIIII": -1.25,
    "IIIIIZ": 0.75,
    "IYIIII": 0.375 + 0.5j,
    "XYZIXI": 2.0 - 1.0j,
    "YYIIZZ": -0.625j,
    "ZXYZIX": 1.5,
}

_KNOWN_3Q = {
    "III": 0.25,
    "XYZ": 1.0 + 2.0j,
    "YIX": -0.5,
    "ZZI": 3.0j,
    "IYI": -1.5 + 0.25j,
}


def _matrix(known):
    dim = 2 ** len(next(iter(known)))
    matrix = np.zeros((dim, dim), dtype=complex)
    for label, coeff in known.items():
        factor = np.ones((1, 1), dtype=complex)
        for letter in label:
            factor = np.kron(factor, _PAULI[letter])
        matrix += coeff * factor
    return matrix


def _assert_exact(terms, known):
    assert set(terms) == set(known), (
        f"extra={sorted(set(terms) - set(known))} "
        f"missing={sorted(set(known) - set(terms))}"
    )
    for label, coeff in known.items():
        assert abs(terms[label] - coeff) < 1e-12, label


def test_sparse_fast_path_recovers_known_coefficients():
    sp = pytest.importorskip("scipy.sparse")
    n_qubits = 6
    matrix = sp.csr_matrix(_matrix(_KNOWN_6Q))

    terms = {}
    for x, z, coeff in parallel_decompose_arrays(
        matrix,
        chunk_size=8,
        n_workers=1,
        executor="thread",
        assume_hermitian=False,
    ):
        terms.update(
            terms_from_arrays(x, z, coeff, n_qubits, assume_hermitian=False)
        )

    _assert_exact(terms, _KNOWN_6Q)


def test_dense_recovers_known_coefficients():
    terms = fwht_pauli_terms(_matrix(_KNOWN_3Q), assume_hermitian=False)
    _assert_exact(terms, _KNOWN_3Q)
