# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""paulikit: Pauli-basis decomposition of large operators.

Exact Pauli decomposition of arbitrary complex matrices, built for
the regime where materialising the full 4^n coefficient set is the
binding constraint rather than the transform itself.

The streaming entry points - ``fwht_pauli_terms_iter``,
``parallel_decompose`` and ``parallel_decompose_arrays`` - yield one
chunk at a time, so peak resident memory is bounded by the chunk size
rather than by the term count. The others return a complete result
and are bounded by it; pick accordingly.

Hermitian and non-Hermitian input are equally supported and take the
same transform. Declaring input Hermitian returns real rather than
complex coefficients, and checks that assumption rather than trusting
it.

The decomposition uses the Fast Walsh-Hadamard Transform, O(N^2 log N)
for an N x N matrix. Several parts have optional compiled kernels that
fall back to NumPy when they are not built, so the package works with
no compiler present. See ``paulikit.algorithms``.

Public API:

Decomposition, from ``paulikit.algorithms.fwht``:

    auto_decompose              pick a strategy from the operator and
                                the machine's available memory
    fwht_pauli_terms            label -> coefficient dict
    fwht_pauli_coefficients     coefficients as a dense array, an
                                active-row block, or COO triples
    fwht_pauli_terms_iter       as fwht_pauli_terms, one chunk at a
                                time
    parallel_decompose          multi-worker, yielding dicts
    parallel_decompose_arrays   multi-worker, yielding raw (x, z,
                                coeff) arrays - no label construction
    terms_from_arrays           render labels for a chosen subset of
                                those arrays
    iter_checkpoint_chunks      read PKCP frames written by
                                checkpoint_path= / CLI --write-chunks

Input preparation, from ``paulikit.hamiltonian``:

    build_hamiltonian           a coupled-oscillator Hamiltonian
    pad_to_power_of_two         pad an operator to a 2^n dimension

Reconstruction and checking, from ``paulikit.pauli_utils``:

    pauli_string_to_matrix      a Pauli string as a dense matrix
    reconstruct_from_terms      rebuild an operator from its terms

``paulikit.testing.fixtures`` additionally provides small worked
decompositions used by the test suite.
"""

__version__ = "0.1.3"
