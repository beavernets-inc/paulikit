# paulikit

Exact Pauli decomposition of arbitrary complex matrices, at scales
where materialising the full coefficient set is the binding
constraint.

Any $2^n \times 2^n$ complex matrix can be written as a weighted sum
over the $4^n$ $n$-qubit Pauli strings. That decomposition is what
turns a Hamiltonian into something a quantum algorithm can consume —
it is the input to linear-combination-of-unitaries (LCU) routines,
to Hamiltonian simulation, and to variational methods. Computing it
is a fast Walsh-Hadamard transform, which is well established and
cheap in theory.

The difficulty is not the transform. It is that the output has $4^n$
entries: at 15 qubits a dense decomposition is over a billion
coefficients, and implementations that build the result in memory
before returning it run out of memory long before they run out of
time.

`paulikit` addresses that directly:

- **Streaming output.** Peak resident memory is bounded by the chunk
  size, not by the term count, so it stays roughly flat as the problem
  grows: measured peak resident set is 72 MiB for a 14-qubit
  decomposition yielding 91,652,096 terms, 89 MiB at 15 qubits
  (326,134,272 terms) and 123 MiB at 16 qubits (1,470,021,632 terms).
  An implementation requiring the caller to hold the dense
  $2^n \times 2^n$ operator needs 4 GiB, 16 GiB and 64 GiB
  respectively for the same three sizes.
- **Exhaustive verification.** Every term is checked individually —
  not sampled — against an independently derived projection oracle.
- **Checkpoint and restart / stream writer.** A binary chunk-framed
  PKCP sink, cheap enough to leave enabled, written only from the main
  drain thread. CLI: `--write-chunks PATH` (alias `--checkpoint-path`).
  Without it, large CLI runs count and discard; with it, results stay
  on disk as `(x, z, coeff)` frames readable via `iter_checkpoint_chunks`.
- **Multi-core execution, in the shipped package.** Decomposition
  runs across multiple workers, selectable per call and from the
  command line: a thread pool that runs the compiled kernels
  concurrently, or a process pool. Chunks are independent by
  construction, and chunk sizing is tuned against measured cache
  boundaries. Scaling is characterised under a documented
  measurement protocol rather than asserted.

Hermitian and non-Hermitian input are equally supported and take the
same transform — neither is a degraded path. `assume_hermitian=True`
(the default) additionally returns real rather than complex
coefficients, and *checks* that the input really is Hermitian rather
than trusting it; pass `assume_hermitian=False` for general complex
matrices. Both cases are separately verified.

Requires Python >= 3.10; the only runtime dependency is NumPy.


## Documentation

| | |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Full build and install reference |
| [`docs/tutorial.md`](docs/tutorial.md) | Step-by-step walkthrough |
| [`docs/runtime_estimates.md`](docs/runtime_estimates.md) | When large runs take seconds vs hours |
| [`docs/theory.md`](docs/theory.md) | Mathematical derivation |
| [`docs/background.md`](docs/background.md) | Physical motivation |
| [`docs/non_hermitian.md`](docs/non_hermitian.md) | Non-Hermitian operators |
| [`docs/package_layout.md`](docs/package_layout.md) | Annotated source tree |


## Installation

**From PyPI** (Linux manylinux wheels for CPython 3.10–3.13, x86_64 and
aarch64; other platforms get the sdist and build or use pure Python):

```bash
pip install paulikit
```

The Linux wheels ship the `wht_kernel` set (Walsh–Hadamard, coefficients,
gather, Hermiticity check, plus optional x86-64-v3 twins),
`cache_probe`, and the **serial** `pauli_label_native` kernel (C +
Cython; no oneTBB). The optional oneTBB parallel label fill is
from-source only — e2e `list[str]` labeling is dominated by Python
string construction, so wheels do not vendor TBB for it. macOS,
Windows, and musllinux are not wheel targets.

**From a source checkout** (development):

```bash
./configure && make          # creates a venv, generates a Makefile
make check                   # run the test suite
```

`./configure` prints a capability report (compiler, Cython, oneTBB,
cache hierarchy, NumPy's BLAS backend) and generates a Makefile with
the standard GNU targets. The build optionally compiles native
Cython/C(/C++) kernels; if the toolchain is unavailable it falls
back to pure Python automatically, with a warning the first time that
path runs.

Equivalent manual sequence (same result as `make`):

```bash
python3 -m venv ~/.venvs/paulikit
source ~/.venvs/paulikit/bin/activate
pip install numpy meson-python cython ninja
pip install -e ".[dev]" --no-build-isolation
```

See [`docs/installation.md`](docs/installation.md) for editable-install
sequencing and how to force the native extensions on or off. Canonical
development is on
[Codeberg](https://codeberg.org/beavernets/paulikit); a
[GitHub mirror](https://github.com/beavernets-inc/paulikit) runs the
manylinux wheel workflow.


## Usage

### Fastest paths (start here for real work)

Two recipes cover the measured high-performance configurations.
Both require the compiled `wht_kernel` modules for best results
(`pip install paulikit` on Linux, or `./configure && make` from a
checkout with a C toolchain). Prefer `executor="thread"` (or CLI
`--executor thread` / `auto`) when those kernels are present.

**Sparse / large-N (CLI) — threaded drain, chunk size 2.** This is the
path that scales: labels are not built; peak RSS stays tens of MiB at
sizes a dense matrix cannot hold.

```bash
paulikit decompose --n-oscillators 150 --chunk-size 2 --parallel \
    --executor thread
# optional: --n-workers N   # default = physical cores
# optional: --write-chunks PATH   # persist streamed (x, z, coeff) frames
# optional: --progress            # main-thread chunk ticks on stderr
```

Without `--write-chunks` (or its alias `--checkpoint-path`), the CLI
**counts** terms and **discards** each chunk after the drain — fine for
timing, not a usable archive. With `--write-chunks PATH`, the **main
drain thread** appends binary PKCP frames (symplectic `x`/`z` plus
`complex` coefficients); workers never touch the file. Read them back
with `iter_checkpoint_chunks` (see below). Pauli-string labels are
optional via `terms_from_arrays` on **subsets** — do not convert
billions of terms to Python strings by default.

`--progress` (opt-in) prints throttled `chunks=k/N (p%) … eta=…`
lines on **stderr** from a **separate** drain loop body — when the
flag is off, the quiet path makes no per-chunk progress calls.
Both paths run a short pre-drain chunk-count prep (outside the
reported Decomposition time) so quiet and progress starts share the
same CPU warm-up. Workers and library APIs never emit progress.

**Dense fast path (library) — skip the sparsity scan.** Use when the
operator is a full dense `ndarray` (e.g. random Hermitian). The CLI
does not yet expose `assume_dense`; call the array API directly:

```python
from paulikit.algorithms.fwht import parallel_decompose_arrays

# H: dense complex128 array, shape (2**n, 2**n)
for x, z, coeff in parallel_decompose_arrays(
    H,
    chunk_size=2,
    assume_dense=True,
    n_workers=1,           # or physical-core count for multi-core
    executor="thread",
):
    ...
```

Publication measurements use dense qubits=13 and sparse N=300 with
these same knobs (`chunk_size=2`, `assume_dense=True` on dense,
`executor="thread"` / `auto` when kernels are present). See the
companion measurements deposit for the protocol and frozen numbers.

### Command line (small examples)

Once installed, the `paulikit` console script is available:

```bash
paulikit --help
paulikit decompose --n-oscillators 4 --show-terms
paulikit benchmark --n-oscillators 2 4 8 16 30
paulikit regenerate-fixtures
```

Run `paulikit <subcommand> --help` for full details on each.

### As a library (small example)

```python
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
H_padded, n_qubits = pad_to_power_of_two(H)

terms = fwht_pauli_terms(H_padded)  # {"IXI": -0.556..., "XII": -0.354..., ...}
```

For large operators prefer `parallel_decompose_arrays` (see **Fastest
paths** above) over collecting a full label dict. To keep results on
disk while streaming:

```python
from paulikit.algorithms.fwht import (
    parallel_decompose_arrays,
    iter_checkpoint_chunks,
    terms_from_arrays,
)

path = "run.pkcp"
for x, z, coeff in parallel_decompose_arrays(
    H_padded, chunk_size=2, checkpoint_path=path, executor="thread",
):
    pass  # optional: filter / reduce per chunk here

# Later (or in another process): one frame at a time
for chunk_index, x, z, coeff in iter_checkpoint_chunks(path):
    # symplectic form is usable as-is; labels only if needed:
    # labels = terms_from_arrays(x[:10], z[:10], coeff[:10], n_qubits)
    ...
```


## Package layout

```
src/paulikit/
    hamiltonian.py      Coupled-oscillator Hamiltonian construction
    pauli_utils.py      Pauli-matrix helpers (label <-> matrix)
    algorithms/fwht.py  The decomposition algorithm
    algorithms/autotune.py  Cache-aware chunk sizing
    testing/fixtures.py Known-good operators and expected outputs
    _native/            Optional compiled kernels, pure-Python fallback
    cli.py              Command-line interface
tests/                  Test suite (pytest)
verification/           Exhaustive correctness runs and their artifacts
docs/                   Tutorial, theory, background, installation
```

See [`docs/package_layout.md`](docs/package_layout.md) for the
annotated tree.


## Running the tests

```bash
pytest
# or: make check
```

(from this directory; `pyproject.toml` sets `testpaths = ["tests"]`
and skips `@pytest.mark.slow` by default. The package must be
installed — `pip install -e ".[test]"` — for imports to resolve).
323 default tests; 8 further slow benchmark comparisons.


## Algorithms implemented

### Fast Walsh-Hadamard Transform (FWHT) — `paulikit.algorithms.fwht`

$O(N^2 \log N)$ for an $N \times N$ matrix — equivalently
$O(n \cdot 4^n)$ for $n$ qubits, since $N = 2^n$. Note the two
symbols differ by an exponential: $n$ counts qubits everywhere else
in this file, $N$ is the matrix side.

Decomposition by Walsh-Hadamard transform is established practice
rather than novel — PennyLane and Classiq both use it, and it is
treated at length in
[Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d).
What is original here is the implementation, not the method: the three
steps (XOR-index gather, Walsh-Hadamard transform, phase-factor
multiplication) were re-derived from the symplectic (X/Z)
representation of Pauli operators and checked against a
definition-level brute-force decomposition before being written in
fast form — see `algorithms/fwht.py`'s module docstring for the
derivation. The contribution of this package is making that
computation memory-bounded, checkpointable, and verified at scale.

Unit-level checks on this algorithm (see `tests/test_fwht.py`; the
whole-package correctness evidence is under **Correctness** below):
- Against a from-scratch brute-force reference on random Hermitian
  matrices ($n=1..4$ qubits): exact match to floating-point
  precision.
- Against `testing.fixtures.ALL_FIXTURES` (real coupled-oscillator
  Hamiltonians at $N=2$, $N=4$): exact label-set and coefficient
  match.


## Correctness

`paulikit`'s output is verified three ways:

- **Exhaustive projection.** Every term of a decomposition is checked
  individually against an independently derived projection oracle —
  not sampled — up to 91,652,096 terms at 14 qubits. The oracle
  computes $\operatorname{Tr}(H P^{\dagger}) / \text{dim}$ directly
  from the projection formula, so it shares no code path with the
  transform it checks. Artifacts and method:
  [`verification/`](https://codeberg.org/beavernets/paulikit/src/branch/main/verification).
- **Cross-implementation.** Where PennyLane's `qml.pauli_decompose`
  can also run, both agree exactly on term count and on coefficients
  within tolerance. PennyLane is a test-only dependency and is never
  imported by `paulikit.algorithms`.
- **Regression suite.** 323 default tests (8 further slow benchmark
  comparisons excluded by default), including crash-recovery and
  checkpoint-format cases.

No performance comparison table is maintained in this README —
hand-maintained numbers go stale. Measured figures, the protocol, and
raw JSON live in the companion **paulikit measurements** Zenodo
deposit (Linux/`perf` reproducibility package), not in this library
repository.


## Status

Alpha. The API is usable and the correctness evidence is strong, but
the version is 0.x and signatures may still change.

Implemented and verified: the FWHT decomposition with optional
compiled kernels (transform, coefficients, gather, Hermiticity check,
labels, cache probe), sparsity-aware coefficients, streaming output
with bounded memory, chunked and parallel execution with cache-aware
auto-tuning, binary checkpoint/restart, and exhaustive verification to
91,652,096 terms at 14 qubits.

Known gaps:

- Published Linux manylinux wheels (x86_64 / aarch64, CPython 3.10–3.13)
  ship `wht_kernel`, `cache_probe`, and serial `pauli_label_native`.
  The optional oneTBB parallel label module is not in the wheel.
  macOS, Windows, and musllinux are not wheel targets — use the sdist
  / from-source build there. Wheel CI runs on the GitHub mirror
  ([github.com/beavernets-inc/paulikit](https://github.com/beavernets-inc/paulikit));
  Codeberg is the canonical tree.
- CPU pinning and topology detection are Linux-only, with a documented
  fallback elsewhere; the non-Linux paths are not yet exercised in CI.


## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
