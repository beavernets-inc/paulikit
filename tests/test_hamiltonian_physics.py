# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Physics tests for ``build_hamiltonian``: the matrix must reproduce
Newton's equations for the coupled oscillators it is built from.

Every reference here comes from Newton's equations,

    m_j x_j'' = sum_{k != j} kappa_jk (x_k - x_j) - kappa_jj x_j,

never from the matrix under test, so a wrong construction cannot pass
by agreeing with itself. The encoding is Babbush, Berry, Kothari,
Somma, Wiebe, "Exponential quantum speedup in simulating coupled
classical oscillators", Phys. Rev. X 13, 041041 (2023),
arXiv:2303.13012, Sec. III: H = -[[0, B], [B^T, 0]] with
B B^T = A = M^{-1/2} F M^{-1/2}, where F is the stiffness matrix
(f_jj = sum_k kappa_jk, f_jk = -kappa_jk).
"""

import numpy as np
import pytest

from paulikit.hamiltonian import build_hamiltonian


def _random_system(n_oscillators, seed):
    rng = np.random.default_rng(seed)
    spring_constants = {
        (i, j): float(rng.uniform(0.2, 2.0))
        for i in range(n_oscillators)
        for j in range(i, n_oscillators)
    }
    masses = [float(m) for m in rng.uniform(0.5, 2.5, n_oscillators)]
    return spring_constants, masses


def _stiffness(n_oscillators, spring_constants):
    """F from Newton's equations: F x is minus the force vector."""
    stiffness = np.zeros((n_oscillators, n_oscillators))
    for (i, j), kappa in spring_constants.items():
        stiffness[i, i] += kappa
        if i != j:
            stiffness[j, j] += kappa
            stiffness[i, j] -= kappa
            stiffness[j, i] -= kappa
    return stiffness


def _mass_weighted_stiffness(n_oscillators, spring_constants, masses):
    inv_sqrt_m = np.diag(1.0 / np.sqrt(masses))
    return inv_sqrt_m @ _stiffness(n_oscillators, spring_constants) @ inv_sqrt_m


def _dense(n_oscillators, spring_constants, masses, sparse):
    hamiltonian = build_hamiltonian(
        n_oscillators, spring_constants, masses, sparse=sparse
    )
    return hamiltonian.toarray() if sparse else hamiltonian


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def sparse(request):
    if request.param:
        pytest.importorskip("scipy.sparse")
    return request.param


@pytest.mark.parametrize("n_oscillators", [2, 3, 4, 7])
def test_b_squared_is_mass_weighted_stiffness(n_oscillators, sparse):
    """The upper-left block of H**2 is B B^T, which must equal A for
    the velocity amplitudes to obey Newton's equations."""
    spring_constants, masses = _random_system(n_oscillators, seed=n_oscillators)
    hamiltonian = _dense(n_oscillators, spring_constants, masses, sparse)
    b_block = -hamiltonian[:n_oscillators, n_oscillators:]

    expected = _mass_weighted_stiffness(n_oscillators, spring_constants, masses)
    assert np.max(np.abs(b_block @ b_block.T - expected)) < 1e-12


@pytest.mark.parametrize("n_oscillators", [3, 6])
def test_spectrum_matches_normal_mode_frequencies(n_oscillators):
    """Basis-independent: the positive eigenvalues of H are the normal-
    mode angular frequencies, whatever encoding or readout is used."""
    spring_constants, masses = _random_system(n_oscillators, seed=10 + n_oscillators)
    hamiltonian = build_hamiltonian(n_oscillators, spring_constants, masses)

    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    positive = np.sort(eigenvalues[eigenvalues > 1e-9])
    expected = np.sort(
        np.sqrt(
            np.linalg.eigvalsh(
                _mass_weighted_stiffness(n_oscillators, spring_constants, masses)
            )
        )
    )
    assert positive.shape == expected.shape
    assert np.max(np.abs(positive - expected)) < 1e-10


def _encoded_state(n_oscillators, spring_constants, masses, positions, velocities):
    """Babbush et al.'s state, written from its definition rather than
    from B: velocity amplitudes sqrt(m_j) x_j', then i times the spring
    extensions sqrt(k_ii) x_i and sqrt(k_jk) (x_j - x_k), in column order."""
    extensions = [
        np.sqrt(spring_constants[(i, i)]) * positions[i] for i in range(n_oscillators)
    ]
    extensions += [
        np.sqrt(spring_constants[(j, k)]) * (positions[j] - positions[k])
        for j in range(n_oscillators)
        for k in range(j + 1, n_oscillators)
    ]
    return np.concatenate([np.sqrt(masses) * velocities, 1j * np.array(extensions)])


@pytest.mark.parametrize("n_oscillators", [3, 5])
def test_evolution_reproduces_newtonian_dynamics(n_oscillators):
    """Evolve the encoded state under H and compare both blocks -
    velocities and spring extensions - against a direct numerical
    integration of Newton's equations from nonzero x(0) and x'(0)."""
    integrate = pytest.importorskip("scipy.integrate")
    spring_constants, masses = _random_system(n_oscillators, seed=20 + n_oscillators)
    masses_arr = np.array(masses)
    stiffness = _stiffness(n_oscillators, spring_constants)
    rng = np.random.default_rng(99)
    positions_0 = rng.normal(size=n_oscillators)
    velocities_0 = rng.normal(size=n_oscillators)
    t_end = 2.5

    def newton(_, state):
        x, v = state[:n_oscillators], state[n_oscillators:]
        return np.concatenate([v, -(stiffness @ x) / masses_arr])

    solution = integrate.solve_ivp(
        newton,
        (0.0, t_end),
        np.concatenate([positions_0, velocities_0]),
        rtol=1e-12,
        atol=1e-13,
    )
    expected = _encoded_state(
        n_oscillators,
        spring_constants,
        masses_arr,
        solution.y[:n_oscillators, -1],
        solution.y[n_oscillators:, -1],
    )

    hamiltonian = build_hamiltonian(n_oscillators, spring_constants, masses)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    psi_0 = _encoded_state(
        n_oscillators, spring_constants, masses_arr, positions_0, velocities_0
    )
    psi_t = eigenvectors @ (np.exp(-1j * eigenvalues * t_end) * (eigenvectors.T @ psi_0))

    assert np.max(np.abs(psi_t - expected)) < 1e-8


def test_each_coupling_column_touches_only_its_two_oscillators(sparse):
    """Column (j, k) of B is sqrt(kappa_jk) (|j> - |k>) / sqrt(M): it
    may only have entries in rows j and k."""
    n_oscillators = 5
    spring_constants, masses = _random_system(n_oscillators, seed=3)
    hamiltonian = _dense(n_oscillators, spring_constants, masses, sparse)

    pairs = [(j, k) for j in range(n_oscillators) for k in range(j + 1, n_oscillators)]
    for offset, (j, k) in enumerate(pairs):
        column = hamiltonian[:n_oscillators, 2 * n_oscillators + offset]
        assert set(np.nonzero(column)[0]) == {j, k}
        kappa = spring_constants[(j, k)]
        assert column[j] == pytest.approx(-np.sqrt(kappa / masses[j]))
        assert column[k] == pytest.approx(np.sqrt(kappa / masses[k]))


def test_n2_matrix_entries():
    """N=2 has a single coupling, (0, 1); pinned entry by entry so the
    N=2 fixture and every N=2 result stay valid."""
    spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
    masses = [1.0, 2.0]
    expected = np.zeros((5, 5))
    entries = {
        (0, 2): -np.sqrt(1.0 / 1.0),
        (1, 3): -np.sqrt(3.0 / 2.0),
        (0, 4): -np.sqrt(2.0 / 1.0),
        (1, 4): np.sqrt(2.0 / 2.0),
    }
    for (row, col), value in entries.items():
        expected[row, col] = expected[col, row] = value

    assert np.array_equal(build_hamiltonian(2, spring_constants, masses), expected)
