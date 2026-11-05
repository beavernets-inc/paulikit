# Correctness verification: findings

Date: 2026-08-28. Machine: Intel i7-8550U. See `README.md` for how
to reproduce any number below — every one traces to a JSON file
under `results/`.

## 1. Why PennyLane can't verify paulikit at target scale

`qml.pauli_decompose`'s own documented theory section (docstring,
PennyLane 0.45.1): "This method internally uses a generalized
decomposition routine to convert the matrix to a weighted sum of
Pauli words ... in time $O(n\,4^n)$." This is PennyLane's own stated
complexity, not a claim derived from reading its implementation. Its
sparse-input support ("processed natively without converting to
dense format") only avoids materializing the *input* as dense — it
does not reduce the
*output* cost, which is driven by the number of possible Pauli labels
(up to $4^n$), independent of $\operatorname{nnz}(H)$.

Confirmed empirically, not just from the docstring: at $N=20$ (8
qubits, $\text{dim}=256$), the same sparse Hamiltonian with
$\operatorname{nnz}(H)=800$ costs PennyLane distinctly more to
decompose in the 49,024-term non-Hermitian case than in the
24,448-term Hermitian case — direct evidence that its cost tracks the
number of *output* terms, not $\operatorname{nnz}(H)$ (see
`results/N20_hermitian_20260828.json` and
`results/N20_nonhermitian_20260828.json`). That output count grows
combinatorially with qubit count regardless of how sparse $H$ is, so
it becomes impractical for PennyLane to keep up well before
paulikit's own target scale — a hard mathematical wall (the *output*
is exponentially large), not an implementation inefficiency. This is
why runs at $N=50$ and above use the projection oracle alone.

Separately confirmed: `qml.pauli_decompose(H, check_hermitian=False)`
correctly decomposes non-Hermitian operators too (reconstructs to
1.57e-16, matching Hermitian-case precision) — the Hermitian check is
an opt-out safety gate, not a real capability limitation. Label
conventions match paulikit's exactly (leftmost char = qubit 0, same
as `paulikit.pauli_utils.pauli_string_to_matrix` and
`paulikit.algorithms.fwht.pauli_label`'s own docstrings).

## 2. The independent method: exhaustive per-term projection

For a Pauli label with symplectic bitmasks `(x_mask, z_mask)`
(convention: leftmost char = qubit 0, bit position $n-1-j$ for qubit
$j$ — matching `pauli_label`'s own docstring exactly, cross-checked
directly against the source, not just inferred from agreement):

$$c = \frac{\operatorname{Tr}(H P_{\text{label}}^{\dagger})}{\text{dim}}$$

computed without ever materializing the full `(dim, dim)` matrix
`P_label`, using: `H[r, c]` contributes only where `r ^ c == x_mask`
(the X-part flips exactly the `x_mask` bits going from row to
column), with value `sign * phase` where

$$\text{sign} = (-1)^{\operatorname{popcount}(r \,\&\, z_{\text{mask}})}
\qquad
\text{phase} = i^{\operatorname{popcount}(x_{\text{mask}} \,\&\, z_{\text{mask}})}$$

This is a direct definitional projection, mathematically independent
of paulikit's own FWHT-based algorithm — not a re-derivation of the
same shortcut.

## 3. Iteration history

| Version | Approach | Verdict |
|---|---|---|
| v1 | `sp.kron`-rebuild `P_label` per label | far too slow to reach $N=150$ |
| v2 | Per-label Python loop over $H$'s nonzeros directly | still too slow |
| v3 | Python-dict bucket $H$'s nonzeros by `r^c`, Python loop per label over its bucket | **did not scale**: timed out at $N=80$ once `nnz` grew — the per-bucket Python loop still scales with bucket size, not $O(1)$ |
| v4 | Fully vectorized NumPy: sort $H$'s nonzeros by `r^c`, sort labels by `x_mask`, `np.searchsorted` to slice each label-group's matching range, grouped broadcast — **no Python-level loop over labels or nonzeros** | scales correctly, see results below — **final method**, implemented in `exhaustive_projection.py` |

A method whose cost is dominated by a Python-level loop cannot serve
as the verification oracle for a package whose whole claim is scale,
so v3's per-bucket loop was not a candidate for a larger timeout
budget — only full vectorization removes the scaling problem.

`np.unique(..., return_index=True, return_inverse=True)` returns
`(unique, index, inverse)` in that fixed order regardless of kwarg
order. An early implementation of v4 swapped `index`/`inverse` on
unpacking, which silently produced wrong coefficients
(`max_abs_error` of 0.016 instead of ~1e-18) while still running
without error. Rerunning the $N=50$ case caught the mismatch.

## 4. Measured results

All values below are from `results/*.json` (see file names for exact
provenance). All errors are floating-point noise floor (~1e-17 to
1e-20), not approximation — this is an exact independent verification,
not a heuristic.

**Note (0.1.4):** this table and §5 were measured on the Hamiltonian
`build_hamiltonian` produced before 0.1.4, which attached every
coupling spring to oscillators 0 and 1. The verification method is
unaffected: it checks the decomposition of whatever matrix it is
given. Re-run on the corrected Hamiltonian (`pytest -m slow
tests/test_exhaustive_verification.py`): N=50 1,231,872 terms
(max abs error 1.04e-17), N=80 6,346,752 (2.78e-17), N=100
19,910,656 (6.94e-18), N=150 90,431,488 (1.04e-17), all passing.

| N (oscillators) | qubits | dim | H.nnz | # terms | method | max abs error | passed |
|---|---|---|---|---|---|---|---|
| 20 | 8 | 256 | 800 | 24,448 | projection + PennyLane (dual oracle) | 1.39e-17 | yes, both |
| 20 (non-Hermitian) | 8 | 256 | 800 | 49,024 | projection + PennyLane (dual oracle) | 1.39e-17 | yes, both |
| 50 | 11 | 2048 | 5,000 | 1,261,568 | projection only | 3.47e-18 | yes |
| 80 | 12 | 4096 | 12,800 | 6,473,728 | projection only | 1.04e-17 | yes |
| 100 | 13 | 8192 | 20,000 | 20,299,776 | projection only | 2.60e-18 | yes |
| 150 | 14 | 16384 | 45,000 | 91,652,096 | projection only (streaming) | 8.67e-19 | yes |

$N=150$ required the streaming path (`--streaming --chunk-size 256`)
— see §5 below for why the non-streaming path cannot succeed here at
any memory size.

## 5. $N=150$'s memory ceiling

At $N=150$, the dict-returning `fwht_pauli_terms` API cannot
complete regardless of available RAM: it is a property of the
package's own API contract, not a defect in the verification method.
Reaching a working run took the following steps:

1. `fwht_pauli_terms(padded)` on a *dense* Hamiltonian, no
   `chunk_size`: OOM-killed.
2. Adding a memory-bounded broadcast cap inside `project_labels`
   (real, worthwhile hardening, kept) and retrying the same
   dense/unchunked call: still OOM-killed — the cap addresses a
   different, smaller problem than the actual one.
3. Instrumenting stage-by-stage under `ulimit -v 12000000` (a hard
   virtual-memory cap, so failures raise `MemoryError` cleanly
   instead of thrashing the whole machine's swap) isolates the
   *actual* failure point: `fwht_pauli_terms`'s own internal
   `_pauli_label_batch` call, building the label-string dict. This is
   a known, already-documented ceiling: the dict-returning
   `fwht_pauli_terms` API cannot complete at $N=150$ *regardless of
   available RAM*, because it re-fuses every chunk's terms into one
   ~91.65M-entry dict before returning. `fwht_pauli_terms_iter`, a
   generator yielding one chunk's terms at a time and never holding
   the combined result, is the existing fix for exactly this.
4. Rewriting the large-N path to consume `fwht_pauli_terms_iter` and
   verify each chunk immediately via `verify_terms_streaming`
   (accumulates only running summary statistics — `n_terms`,
   `max_abs_error`, `worst_label` — never the term dict itself), and
   switching Hamiltonian construction to
   `build_hamiltonian(..., sparse=True)` +
   `pad_to_power_of_two(..., sparse=True)` end-to-end to avoid ever
   materializing paulikit's own documented ~4 GiB dense-matrix cost
   at this $N$, run under `ulimit -v 12000000` for safety, succeeds:
   91,652,096 terms, max_abs_error=8.67e-19.

`fwht_pauli_terms` (no streaming) is fine through at least $N=100$
(20.3M terms, confirmed working); $N=150$ requires
`fwht_pauli_terms_iter` unconditionally, independent of how much RAM
is available — not a "nice to have for lower memory" option at this
scale.

## 6. Method notes

- Sampling is not used. Every term paulikit outputs is checked
  individually, at every $N$ tested.
- The non-Hermitian dual-oracle check (§4, $N=20$ row) uses a
  deterministic antisymmetric-imaginary perturbation on the real
  Hamiltonian's own sparsity pattern (`run_verification.py`'s
  `make_non_hermitian`, seeded, default seed 42) — not an arbitrary
  random matrix — so it stays a physically meaningful test case, for
  example a non-Hermitian effective Hamiltonian, rather than noise.
