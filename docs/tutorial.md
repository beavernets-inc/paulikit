# Tutorial

This page walks through using `paulikit` end to end: building a
Hamiltonian, decomposing it, interpreting and verifying the result,
and using the command-line interface. For the "why" behind each step,
see {doc}`background` and {doc}`theory`; for the full function
signatures, see the {doc}`API reference <api/index>`.

## Installation

```bash
pip install paulikit
```

On Linux that installs a manylinux wheel with the `wht_kernel` modules
(and `cache_probe`). From a source checkout (development, or platforms
without a wheel):

```bash
./configure && make
```

Or the equivalent editable install:

```bash
pip install numpy meson-python cython ninja
pip install -e . --no-build-isolation
```

Everything in this tutorial works with just the core install (`numpy`
is the only runtime dependency). The command-line examples below
assume the `paulikit` console script is on your `PATH`, which the
install above sets up automatically.

The build optionally compiles several Cython kernels (Walsh–Hadamard
butterfly, coefficients, gather, Hermiticity check, Pauli labels,
cache probe) when a C/C++ toolchain is available — falling back to
pure Python / NumPy automatically otherwise. Published Linux wheels
already include the transform kernels; see {doc}`installation` for
meson options. Nothing in the small examples below depends on
which path is active, but the large-scale recipes in
[§7 Fastest paths](#fastest-paths) do.

## 1. Building a Hamiltonian

`paulikit.hamiltonian.build_hamiltonian` constructs the
coupled-oscillator Hamiltonian matrix from physical parameters: a
dictionary of spring constants and a list of masses.

```python
from paulikit.hamiltonian import build_hamiltonian

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
```

`spring_constants` maps `(i, j)` with `i <= j` to the spring constant
$k_{ij}$: diagonal entries are each oscillator's own spring constant
(coupling to a fixed wall), off-diagonal entries are the coupling
strength between oscillator `i` and oscillator `j`. `masses[i]` is
oscillator `i`'s mass.

The result is a $5 \times 5$ matrix (for $N=2$, the dimension is
$N + N(N+1)/2$):

```
[[ 0.          0.         -1.          0.         -1.41421356]
 [ 0.          0.          0.         -1.22474487  1.        ]
 [-1.          0.          0.          0.          0.        ]
 [ 0.         -1.22474487  0.          0.          0.        ]
 [-1.41421356  1.          0.          0.          0.        ]]
```

## 2. Padding to a power-of-two dimension

Pauli decomposition requires the matrix dimension to be exactly
$2^n$ for some integer $n$ (the number of qubits). Real
coupled-oscillator Hamiltonians rarely have a power-of-two dimension
already, so pad first:

```python
from paulikit.hamiltonian import pad_to_power_of_two

H_padded, n_qubits = pad_to_power_of_two(H)
# H_padded.shape == (8, 8), n_qubits == 3
```

This zero-pads $H$ into the top-left block of an $8 \times 8$ matrix
(since $\lceil \log_2 5 \rceil = 3$). The padding entries are all
zero, so they contribute nothing to the physics — they exist only to
satisfy the tensor-product structure quantum circuits require.

## 3. Decomposing into Pauli terms

```python
from paulikit.algorithms.fwht import fwht_pauli_terms

terms = fwht_pauli_terms(H_padded)
```

`terms` is a dictionary mapping Pauli-string labels to their
coefficients:

```
{
    'IXI': -0.556186, 'IXZ': 0.056186,
    'XII': -0.353553, 'XIX': 0.250000, 'XIZ': -0.353553,
    'XZI': -0.353553, 'XZX': 0.250000, 'XZZ': -0.353553,
    'YIY': 0.250000, 'YZY': 0.250000,
    'ZXI': -0.556186, 'ZXZ': 0.056186,
}
```

Each label is a string of length `n_qubits`, read left-to-right as
qubit 0, 1, 2, ... — e.g. `'XIZ'` means $X$ on qubit 0, $I$
(identity) on qubit 1, $Z$ on qubit 2, i.e. the operator
$X \otimes I \otimes Z$. Only nonzero terms are included (12 of the
$4^3 = 64$ possible 3-qubit Pauli strings, here); the threshold is
controlled by `fwht_pauli_terms`'s `atol` parameter.

By default (`assume_hermitian=True`), coefficients are returned as
real `float`s, and a `ValueError` is raised if the input wasn't
actually Hermitian (a useful sanity check — see {doc}`non_hermitian`
for when and how to decompose non-Hermitian operators instead).

## 4. Verifying the result

`paulikit.pauli_utils.reconstruct_from_terms` rebuilds the dense
matrix from a term dictionary, useful both as a sanity check and for
programmatically confirming a decomposition is correct:

```python
from paulikit.pauli_utils import reconstruct_from_terms

reconstructed = reconstruct_from_terms(terms, n_qubits)
error = (reconstructed.real - H_padded)
print(abs(error).max())  # 0.0 - exact to floating-point precision
```

This is exactly the check `paulikit`'s own test suite runs against
every fixture (see `tests/test_fwht.py`), and it's good practice to
run it yourself whenever decomposing a new Hamiltonian you haven't
validated before.

## 5. Using the command-line interface

For quick exploration without writing a script, the `paulikit`
console command wraps the same functionality:

```console
$ paulikit decompose --n-oscillators 4 --show-terms
N=4 oscillators, 4 qubits, 16x16 padded Hamiltonian
Decomposition time: ...
Nonzero Pauli terms: 56
  IXII: -0.5470915958155509
  IXIZ: 0.015053558406667805
  IXZI: 0.02983035390312644
  ...
```

(Timing is reported but not reproducible across machines; term count
and coefficient values are the part worth checking against your own
run.)

By default, `paulikit decompose` builds a synthetic Hamiltonian
internally (a fixed, deterministic — not physically calibrated — set
of spring constants and masses that scale with $N$), so that mode is
meant for quickly checking behavior and timing at a given size, not
for physically meaningful results. Pass `--operator-file` to stream
your own dense on-disk operator instead (see {doc}`dense_out_of_core`),
or use the library API (above) with your own parameters.

`paulikit benchmark` sweeps multiple $N$ values and reports timing:

```console
$ paulikit benchmark --n-oscillators 2 4 8 16 30
    N  qubits    dim    terms   time (s)
    2       3      8       12        ...
    4       4     16       56        ...
    8       6     64      928        ...
   16       8    256    15360        ...
   30       9    512   112384        ...
```

(Timings vary run to run and by machine; term counts are the part
worth checking against your own run.)

### Decomposing in parallel from the command line

`--parallel` runs the decomposition across multiple workers through
`parallel_decompose_arrays`, which is the path that scales. It needs
`--chunk-size` (the CLI does not auto-tune this flag; pick a value —
`2` is the publication default for sparse large-$N$ work):

```console
$ paulikit decompose --n-oscillators 150 --chunk-size 2 --parallel \
      --executor thread
N=150 oscillators, 14 qubits, 16384x16384 padded Hamiltonian
Decomposition time (parallel, executor=thread): ...
Chunks: 5595, nonzero Pauli terms: 91652096
```

Related flags for the same large-$N$ / file-backed regime:

- `--operator-file PATH` — dense on-disk operator (raw `complex128` +
  sidecar JSON, or square `complex128` `.npy`) instead of the
  synthetic Hamiltonian. Requires `--parallel` and `--chunk-size`.
  Companion flags: `--operator-meta`, `--spill-dir`,
  `--max-resident-buckets`. Details: {doc}`dense_out_of_core`.
- `--write-chunks PATH` — write each completed chunk as binary PKCP
  frames (symplectic `x`, `z`, `coeff`) from the **main drain thread
  only**. Without this flag the CLI counts terms and discards chunk
  arrays. Alias: `--checkpoint-path` (same format; enables resume).
  Read with `paulikit.algorithms.fwht.iter_checkpoint_chunks`.
- `--progress` — opt-in chunk progress on **stderr** (`k/N`, percent,
  ETA) from a separate main-drain loop body (quiet path has no
  per-chunk progress calls). Off by default. Parallel/stream CLI
  paths always pre-count chunks before the timed drain so quiet and
  progress runs share the same CPU warm-up. Library APIs never emit
  progress.
- `--stream` — sequential chunked streaming via `fwht_pauli_terms_iter`
  (labels per chunk; requires `--chunk-size`). Use when you want
  labelled dicts without the multi-core drain. Not combinable with
  `--operator-file` (use `--parallel` for file-backed dense).

For when discard vs write vs materialising a full dict turns into
minutes or hours on modest hardware, see
{doc}`runtime_estimates`.

Two things this path does differently, both deliberate:

- **The operator is built sparse.** At 15 qubits a dense operator is
  16 GiB and at 16 qubits it is 64 GiB, so densifying would put the
  sizes this path exists for out of reach before the decomposition
  started.
- **Labels are never built.** That per-term Python work is the serial
  cost this API removes, so `--show-terms` reports counts only and
  says so rather than silently ignoring the flag. Use
  `terms_from_arrays` in a script if you need labels for a subset.

`--executor` chooses how chunks are drained:

| value | behaviour |
|---|---|
| `auto` (default) | `thread` when the compiled kernels are available, `process` otherwise |
| `thread` | one process, threads running the kernels concurrently — no pickling |
| `process` | a process pool; pays IPC but does not depend on the kernels being built |

The default is conditional because the right answer flips depending on
build. The compiled kernels release the GIL, so with them threads win
decisively; without them the NumPy fallback holds the GIL and threads
are slower than processes. Deciding per build rather than globally is
what makes a default safe here.

`--n-workers` sets the worker count, defaulting to the number of
distinct physical cores. Hyperthread siblings share execution units
rather than adding independent ones, so they measured worse: on a
4-core machine, adding the 4 hyperthread siblings on top of the 4
physical-core threads cost extra CPU cycles for no further wall-clock
gain.

With the thread executor, chunks are split into exactly `--n-workers`
contiguous slices up front and each worker drains its own slice
directly — a static partition, not a shared work queue. Measured
directly: fewer instructions, fewer cycles, better instruction-per-
cycle throughput, and far tighter run-to-run variance than submitting
one task per chunk to a shared queue, because it removes that queue's
lock contention from the hot path almost entirely. Has no effect on
the result or on `--executor process`, which still drains one task
per chunk through a process pool.

By default, all `--n-workers` OS threads are also forced to exist
before the first chunk is submitted, instead of `ThreadPoolExecutor`'s
own lazy spin-up (a new thread only on the first `submit()` that finds
none idle). Measured directly: without this, the first several chunks
of a run start staggered by a few hundred microseconds each rather
than together, since each of the pool's first few `submit()` calls on
a fresh pool forces a new OS thread creation. Pass `--no-eager-threads`
to restore the old lazy spin-up — a one-time cost either way, this
only changes *when* it is paid, not whether it is paid, and has no
effect on the result or on `--executor process`.

`paulikit regenerate-fixtures` recomputes the expected Pauli terms
used by the test suite's correctness fixtures, using PennyLane as an
independent oracle — see the {doc}`API reference <api/testing>` for
`paulikit.testing.fixtures` if you're extending the test suite itself
rather than just using the library.

Run `paulikit --help` or `paulikit <subcommand> --help` for full
argument details on any of these.

## 6. Multi-core decomposition and the array-yielding API

> **Start from [§7 Fastest paths](#fastest-paths) for real work.** This
> section explains the two library APIs underneath those recipes.
> Throughput depends on which executor and build you use — see
> "Which executor, and why it matters" below. The array-yielding path
> is always the right choice for streaming and bounded memory,
> independent of throughput.

For large operators the default recommendation is
`parallel_decompose_arrays`: it yields raw `(x, z, coeff)` NumPy
arrays, supports `executor="thread"` / `"process"` / `"auto"`,
auto-tunes `chunk_size` when omitted, and shares the binary
chunk-framed checkpoint format with the labelled API. Use it when you
do not need every Pauli label up front — for instance filtering to
the largest-magnitude terms, or feeding coefficients into a numerical
routine that never looks at the string itself:

```python
import numpy as np
from paulikit.algorithms.fwht import parallel_decompose_arrays, terms_from_arrays

n_qubits = int(np.log2(H_padded.shape[0]))
for x, z, coeff in parallel_decompose_arrays(H_padded):
    big = np.abs(coeff) > 1e-3          # keep only what you need
    terms = terms_from_arrays(x[big], z[big], coeff[big], n_qubits)
```

`terms_from_arrays` is the opt-in rendering step: pass it whichever
arrays (or filtered subset of them) you actually want labels for, and
it returns the same `dict[str, float]` (or `dict[str, complex]` if
`assume_hermitian=False`) that `fwht_pauli_terms` and
`parallel_decompose` produce. Because labeling a handful of surviving
terms is cheap regardless of how large the original decomposition was,
this pattern — decompose with `parallel_decompose_arrays`, filter, then
label only the survivors — sidesteps the serial bottleneck rather than
paying it and discarding most of the result.

How much does this actually buy you? Removing per-term Python work
from the drain loop is a real and well-understood fix for a real
serial bottleneck — the label-and-dict path scaled *negatively*,
where adding workers made it slower.

When you *do* need every term's Pauli label, use
`parallel_decompose`. It still spreads chunk coefficient work across
workers, but always through a process pool, and builds
`dict[str, complex]` (or `float`) labels in the parent process — the
same serial cost the array API removes. Checkpoints are interchangeable
between the two functions (same binary chunk-framed format). Prefer it
only when the labelled dict is the product you want; otherwise stay on
`parallel_decompose_arrays`.

### Which executor, and why it matters

`parallel_decompose_arrays` takes an `executor` argument. The default,
`"auto"`, picks `"thread"` when the compiled kernels are available and
`"process"` otherwise.

That conditional is not hedging: the two builds have opposite winners.
With the compiled kernels present, `"thread"` beats `"process"` by a
wide margin; with the pure-NumPy fallback, `"process"` beats
`"thread"` instead. The mechanism explains why: the kernels release
the GIL, so with them threads run the real work concurrently and pay
no pickling at all. Without them the NumPy fallback holds the GIL
through most of each chunk, so threads serialise *and* add contention.
Neither choice is right in general, which is why it is decided per
build.

At N=150 (91.6M terms), peak RSS stays in the same tens-of-MiB range
across all three paths (sequential chunked, parallel threads, parallel
processes), so the choice among them is a throughput question, not a
memory one. With the compiled kernels present, the parallel-threads
path avoids the process pool's IPC and pickling cost entirely, which
is why it is the path worth reaching for first once the kernels are
built; measure on your own build and machine to confirm the ranking
before relying on it.

### Reaching sizes a dense implementation cannot

The streaming design's real payoff is not the constant factor. Peak
resident memory stays roughly flat as the problem grows, because only
one chunk is live at a time:

| qubits | terms | peak RSS |
|---|---|---|
| 14 | 91,652,096 | 72 MiB |
| 15 | 326,134,272 | 89 MiB |
| 16 | 1,470,021,632 | 123 MiB |

An implementation that requires the caller to hold the dense
$2^n \times 2^n$ operator needs 4 GiB at 14 qubits, 16 GiB at 15 and
64 GiB at 16 — so on a 16 GiB machine the last two are simply out of
reach, regardless of how fast its inner loop is.

Measure your own workload rather than assuming either way — but the
memory profile is a property of the design, not of tuning.

## 7. Fastest paths

Two recipes cover the measured high-performance configurations.
Prefer `executor="thread"` (or CLI `--executor thread` / `auto`) when
the compiled `wht_kernel` modules are present.

**Sparse / large-N (CLI) — threaded drain, chunk size 2.** Labels are
not built; peak RSS stays tens of MiB at sizes a dense matrix cannot
hold:

```bash
paulikit decompose --n-oscillators 150 --chunk-size 2 --parallel \
    --executor thread
# optional: --n-workers N   # default = physical cores
# optional: --write-chunks PATH  # PKCP stream writer (alias --checkpoint-path)
# optional: --progress           # main-thread chunk ticks on stderr
```

**Dense fast path (library) — skip the sparsity scan.** Use when $H$
is already a resident dense `ndarray`. The CLI does not expose
`assume_dense`; call the array API directly:

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

**Dense on disk — step-by-step tutorial below.** When the dense matrix
itself is the memory wall, keep $H$ as a file and stream it; do not
call `np.load` on the full tile. See
[§8 Dense operators on disk](#dense-operators-on-disk) and the reference
page {doc}`dense_out_of_core`.

Publication measurements use dense qubits=13 and sparse $N=300$ with
the resident dense / sparse knobs above. Measured figures and the
protocol live in the companion measurements deposit, not in this
tutorial.

For labelled small operators, keep using `fwht_pauli_terms` as in
§3. For large operators prefer `parallel_decompose_arrays` over
collecting a full label dict.

(dense-operators-on-disk)=
## 8. Dense operators on disk (I/O)

This walkthrough builds a **layout A** dense input (raw row-major
`complex128` + JSON sidecar), runs an out-of-core decomposition that
**spills input buckets** to disk, optionally **writes Pauli results**
as PKCP frames, and reads those frames back. Prefer a **local SSD**
for spill and PKCP paths (network filesystems can dominate runtime).

Full format contract (required sidecar fields, `.npy` layout B,
rejected formats): {doc}`dense_out_of_core`.

### Step 1 — Write the operator file (layout A)

For sizes that fit in RAM while *writing*, build a Hermitian matrix
and dump it with NumPy's `tofile` (binary, no header):

```python
import json
from pathlib import Path
import numpy as np

n_qubits = 4          # demo size; raise carefully — |H| = 16 * 4**n bytes
dim = 2**n_qubits
rng = np.random.default_rng(0)
raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
H = np.ascontiguousarray((raw + raw.conj().T) / 2, dtype=np.complex128)

op_dir = Path("dense_demo")
op_dir.mkdir(exist_ok=True)
blob = op_dir / "H.c128"
H.tofile(blob)

meta = {
    "format": "paulikit.dense_c128.v1",
    "dim": dim,
    "dtype": "complex128",
    "layout": "row-major",
    "endianness": "little",   # must match the host in v1
    "n_qubits": n_qubits,
    "codec": "identity",
}
Path(str(blob) + ".json").write_text(json.dumps(meta, indent=2))
print(blob, "bytes", blob.stat().st_size)  # must equal dim*dim*16
```

**Larger tiles** (when you cannot hold $H$ and $H^\dagger$ while
writing): stream **rows** of `complex128` with `fh.write(row.tobytes())`
instead of materialising the full array. That file is a valid dense
input for the FWHT path; if it is not Hermitian, pass
`assume_hermitian=False` at drain time (see Step 3).

**Layout B alternative:** `np.save("H.npy", H)` with square C-order
`complex128` — no sidecar required. Same decompose steps with
`--operator-file H.npy` / `from_dense_file("H.npy", ...)`.

### Step 2 — Choose directories and `chunk_size`

Three disk roles — do not conflate them:

| Role | Typical path | What it is |
|---|---|---|
| Input blob | `dense_demo/H.c128` (+ `.json`) | Dense $H$ on disk |
| Input spill | `dense_demo/buckets/` | Pass‑1 $x$-buckets (temporary working set) |
| Result archive | `dense_demo/run.pkcp` | Optional PKCP Pauli chunks (output) |

`chunk_size` is both the bucket height and the drain tile size. With
today's Pass‑1 implementation, the matrix is re-read roughly once per
bucket, so **very small** `chunk_size` (e.g. 2) at large $n$ means
huge I/O. For demos, use a moderate value (e.g. 64–256 for tiny $n$;
larger for big files). Prefer `max_resident_buckets=1` or `2` so peak
RAM tracks one bucket, not all of them.

### Step 3 — Decompose from disk (library)

```python
from paulikit.algorithms.dense_bucketed import DenseBucketedSource
from paulikit.algorithms.fwht import (
    parallel_decompose_arrays,
    iter_checkpoint_chunks,
)

blob = "dense_demo/H.c128"
spill = "dense_demo/buckets"
pkcp = "dense_demo/run.pkcp"
chunk_size = 64

src = DenseBucketedSource.from_dense_file(
    blob,
    meta=blob + ".json",       # or omit if PATH.json exists beside the blob
    chunk_size=chunk_size,
    spill_dir=spill,
    max_resident_buckets=1,
)

# Pass-1 has now written spill tiles under spill_dir.
# Drain: no full H in RAM. Optional PKCP result write on the main thread.
total_terms = 0
for x, z, coeff in parallel_decompose_arrays(
    None,
    operator_source=src,
    chunk_size=src.chunk_size,
    n_workers=1,                 # or physical-core count
    executor="thread",
    assume_hermitian=True,       # False if the on-disk tile is not Hermitian
    checkpoint_path=pkcp,        # omit to count/discard only
):
    total_terms += len(coeff)

print("nonzero terms", total_terms)

# Later / another process: stream results without reloading H
for chunk_index, x, z, coeff in iter_checkpoint_chunks(pkcp):
    ...  # filter / reduce; labels via terms_from_arrays on subsets only
```

### Step 4 — Same path from the CLI

```bash
paulikit decompose \
  --operator-file dense_demo/H.c128 \
  --operator-meta dense_demo/H.c128.json \
  --parallel --chunk-size 64 \
  --spill-dir dense_demo/buckets \
  --max-resident-buckets 1 \
  --executor thread --n-workers 1 \
  --write-chunks dense_demo/run.pkcp \
  --progress
```

Requires `--parallel` and `--chunk-size`. Skips the synthetic
oscillator Hamiltonian. `--write-chunks` is the **result** archive;
`--spill-dir` is **input** working space.

### Step 5 — What “success” looks like

- Sidecar validates (`dim` power of two, size `dim*dim*16`, matching
  endianness).
- Spill directory fills with `bucket_XXXXXX.c128` tiles during Pass‑1,
  then the drain prints chunk / term counts (CLI) or your loop finishes
  (library).
- With `--write-chunks` / `checkpoint_path`, `iter_checkpoint_chunks`
  replays symplectic `(x, z, coeff)` frames without touching `H.c128`
  again.

### Step 6 — Common pitfalls

- **RSS while writing $H$:** building a Hermitian tile in RAM can peak
  near $2\cdot|H|$ even though the later OOC drain stays small. Write
  in one process; measure drain in a fresh process if you care about
  peak RSS.
- **`assume_hermitian=True` (default)** on a non-Hermitian file raises
  on imaginary diagonal / identity-term mass — pass `False` or write a
  Hermitian blob.
- **`chunk_size=2` at large $n$:** fine for *resident* dense/sparse
  drains in the recipes above; costly for *current* file-backed Pass‑1
  until a single-pass scatter lands. See {doc}`dense_out_of_core`.
- **Sparse operators:** do not densify into `.c128` — use the sparse
  `--parallel` / CSR path instead.

