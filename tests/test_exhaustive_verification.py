# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Exhaustive, PennyLane-independent correctness check for
``paulikit.algorithms.fwht.fwht_pauli_terms``.

Unlike ``test_benchmark_reference.py``'s
``test_paulikit_vs_pennylane_on_matched_hamiltonian`` (which only
checks term *counts* at large N, since PennyLane's O(n*4^n) decompose
cannot finish there), this module checks every single Pauli
coefficient exactly, at any N, via
``verification.exhaustive_projection`` - a direct projection formula
independent of paulikit's own FWHT algorithm. See
``verification/FINDINGS.md`` for the full design writeup and measured
timings.

Fast cases run in CI; N>=50 cases are marked slow (same convention as
test_benchmark_reference.py) since exhaustive coverage at that scale
takes tens of seconds to minutes.
"""

import sys
from pathlib import Path

import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verification"))

from exhaustive_projection import verify_terms, verify_terms_streaming  # noqa: E402

from paulikit.algorithms.fwht import fwht_pauli_terms, fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _synthetic_spring_constants(n_oscillators):
    return {
        (i, j): 1.0 + 0.1 * (i + j)
        for i in range(n_oscillators)
        for j in range(i, n_oscillators)
    }


def _synthetic_masses(n_oscillators):
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


def _build_padded(n_oscillators, hermitian=True, seed=42):
    spring_constants = _synthetic_spring_constants(n_oscillators)
    masses = _synthetic_masses(n_oscillators)
    unpadded = build_hamiltonian(n_oscillators, spring_constants, masses)
    padded, n_qubits = pad_to_power_of_two(unpadded)
    if not hermitian:
        import numpy as np

        rng = np.random.default_rng(seed)
        mask = padded != 0
        perturbation = np.zeros_like(padded, dtype=complex)
        rows, cols = np.nonzero(np.triu(mask, k=1))
        values = rng.uniform(0.01, 0.05, size=len(rows)) * 1j
        perturbation[rows, cols] = values
        perturbation[cols, rows] = -values
        padded = padded.astype(complex) + perturbation
    return padded, n_qubits


@pytest.mark.parametrize("n_oscillators", [4, 8, 16, 20])
def test_exhaustive_projection_hermitian(n_oscillators):
    padded, n_qubits = _build_padded(n_oscillators, hermitian=True)
    terms = fwht_pauli_terms(padded)
    result = verify_terms(sp.csr_matrix(padded), terms)
    assert result["passed"], (
        f"N={n_oscillators}: max_abs_error={result['max_abs_error']} "
        f"on label {result['worst_label']!r}"
    )


@pytest.mark.parametrize("n_oscillators", [4, 8, 20])
def test_exhaustive_projection_non_hermitian(n_oscillators):
    padded, n_qubits = _build_padded(n_oscillators, hermitian=False)
    terms = fwht_pauli_terms(padded, assume_hermitian=False)
    result = verify_terms(sp.csr_matrix(padded), terms)
    assert result["passed"], (
        f"N={n_oscillators} (non-Hermitian): max_abs_error="
        f"{result['max_abs_error']} on label {result['worst_label']!r}"
    )


@pytest.mark.parametrize("n_oscillators", [20])
def test_dual_oracle_agrees_with_pennylane(n_oscillators):
    """Small-N only: full PennyLane decomposition AND the exhaustive
    projection must both agree with paulikit's output, exactly."""
    qml = pytest.importorskip("pennylane")

    padded, n_qubits = _build_padded(n_oscillators, hermitian=True)
    terms = fwht_pauli_terms(padded)

    proj_result = verify_terms(sp.csr_matrix(padded), terms)
    assert proj_result["passed"]

    # Dense input on purpose: on Python 3.10 with PennyLane 0.42,
    # pauli_decompose fails on a scipy sparse matrix (autoray has no
    # 'gather' for the scipy backend). padded is already dense.
    pl_result = qml.pauli_decompose(padded)
    pl_coeffs, pl_ops = pl_result.terms()
    pl_terms = {}
    for coeff, op in zip(pl_coeffs, pl_ops):
        label_chars = ["I"] * n_qubits
        pauli_rep = op.pauli_rep
        if pauli_rep is not None:
            ((pw, _c),) = pauli_rep.items()
            for wire, letter in pw.items():
                label_chars[wire] = letter
        pl_terms["".join(label_chars)] = complex(coeff)

    assert len(pl_terms) == len(terms)
    max_diff = max(abs(complex(c) - pl_terms[label]) for label, c in terms.items())
    assert max_diff < 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("n_oscillators", [50, 80, 100])
def test_exhaustive_projection_large_n(n_oscillators, capsys):
    padded, n_qubits = _build_padded(n_oscillators, hermitian=True)
    terms = fwht_pauli_terms(padded)
    result = verify_terms(sp.csr_matrix(padded), terms)

    with capsys.disabled():
        print(
            f"\nN={n_oscillators} n_terms={result['n_terms']} "
            f"max_abs_error={result['max_abs_error']:.3e}"
        )

    assert result["passed"], (
        f"N={n_oscillators}: max_abs_error={result['max_abs_error']} "
        f"on label {result['worst_label']!r}"
    )


@pytest.mark.slow
def test_exhaustive_projection_n150_streaming(capsys):
    """N=150 requires the streaming API unconditionally, independent of
    available RAM - see verification/FINDINGS.md section 6 for why the
    dict-returning fwht_pauli_terms cannot complete at this scale at
    all. ~6 minutes; run explicitly with
    `pytest -m slow -k n150 -s`."""
    spring_constants = _synthetic_spring_constants(150)
    masses = _synthetic_masses(150)
    H_sparse = build_hamiltonian(150, spring_constants, masses, sparse=True)
    padded_sparse, n_qubits = pad_to_power_of_two(H_sparse, sparse=True)
    padded_sparse = padded_sparse.tocsr()

    chunk_iter = fwht_pauli_terms_iter(padded_sparse, chunk_size=256)
    result = verify_terms_streaming(padded_sparse, chunk_iter)

    with capsys.disabled():
        print(
            f"\nN=150 n_terms={result['n_terms']} "
            f"max_abs_error={result['max_abs_error']:.3e}"
        )

    assert result["n_terms"] == 90_431_488
    assert result["passed"], (
        f"max_abs_error={result['max_abs_error']} on label "
        f"{result['worst_label']!r}"
    )
