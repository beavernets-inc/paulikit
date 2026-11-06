# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Not a correctness test - a documented, reproducible comparison
between paulikit.algorithms.fwht and PennyLane's qml.pauli_decompose
on the *same* real coupled-oscillator Hamiltonians, at matched N.

Skipped by default (marked slow) since N=50/100 take PennyLane
minutes; run explicitly with:

    pytest tests/test_benchmark_reference.py -m slow -s

Requires the `test` extra (PennyLane) to be installed.

Why this exists: an earlier baseline comparison used a synthetic
tridiagonal-band sparse matrix as a stand-in for "the real
Hamiltonian's sparsity pattern," rather than the real
build_hamiltonian() output. That stand-in happened to decompose to
far fewer Pauli terms than the actual coupled-oscillator Hamiltonian
does at the same N, which understated PennyLane's real runtime and
made the eventual comparison look more lopsided in paulikit's favor
than it actually is. This test/script benchmarks both implementations
against the exact same matrix, so the numbers are directly comparable.
"""

import time

import numpy as np
import pytest

from paulikit.algorithms.fwht import fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _synthetic_spring_constants(n_oscillators):
    constants = {}
    for i in range(n_oscillators):
        for j in range(i, n_oscillators):
            constants[(i, j)] = 1.0 + 0.1 * (i + j)
    return constants


def _synthetic_masses(n_oscillators):
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


def _build_padded(n_oscillators):
    spring_constants = _synthetic_spring_constants(n_oscillators)
    masses = _synthetic_masses(n_oscillators)
    unpadded = build_hamiltonian(n_oscillators, spring_constants, masses)
    return pad_to_power_of_two(unpadded)


@pytest.mark.slow
@pytest.mark.parametrize("n_oscillators", [16, 30, 50, 100])
def test_paulikit_vs_pennylane_on_matched_hamiltonian(n_oscillators, capsys):
    """Prints a timing/term-count comparison; does not assert on timing
    (machine-dependent), only on term-count and coefficient agreement,
    which is the actual correctness claim worth automating."""
    qml = pytest.importorskip("pennylane")

    padded, n_qubits = _build_padded(n_oscillators)

    start = time.perf_counter()
    paulikit_terms = fwht_pauli_terms(padded)
    paulikit_time = time.perf_counter() - start

    start = time.perf_counter()
    # Dense input: pauli_decompose fails on a scipy sparse matrix on
    # Python 3.10 with PennyLane 0.42 (autoray has no 'gather').
    pennylane_result = qml.pauli_decompose(padded)
    pennylane_time = time.perf_counter() - start
    pennylane_coeffs, _ = pennylane_result.terms()

    with capsys.disabled():
        print(
            f"\nN={n_oscillators} qubits={n_qubits} "
            f"paulikit={paulikit_time:.4f}s ({len(paulikit_terms)} terms) "
            f"pennylane={pennylane_time:.4f}s ({len(pennylane_coeffs)} terms) "
            f"speedup={pennylane_time / paulikit_time:.1f}x"
        )

    assert len(paulikit_terms) == len(pennylane_coeffs), (
        f"N={n_oscillators}: term count mismatch - "
        f"paulikit={len(paulikit_terms)}, pennylane={len(pennylane_coeffs)}"
    )
