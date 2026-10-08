# Dense operators that do not fit in RAM

Peak resident memory during a Pauli drain is usually set by
**chunk size**, not by the number of Pauli terms. A second wall
appears when the **input** operator is dense: a full
$2^n \times 2^n$ `complex128` matrix needs $16 \cdot 4^n$ bytes
(about 4 GiB / 16 GiB / 64 GiB at $n = 14,15,16$). Streaming the
*output* does not help if you first allocate that matrix in RAM.

This page documents the **opt-in** out-of-core dense path: keep $H$
on disk, stream rows into fixed-size $x$-buckets (with optional
spill), and feed the same `gather_chunk` → WHT → drain pipeline used
for resident dense and sparse work.

Sparse operators should stay on the existing CSR /
`parallel_decompose_arrays` sparse path — do not densify them into
this format.

## When to use it

| Situation | Path |
|---|---|
| Sparse / structured Hamiltonian | Sparse CLI or library (default `--parallel` demo) |
| Dense $H$ that fits in RAM | `parallel_decompose_arrays(H, assume_dense=True, …)` |
| Dense $H$ on disk (too large to hold) | **This page** — `from_dense_file` / `--operator-file` |
| Already have Pauli terms | Do not densify; consume or convert terms directly |

## On-disk layouts (v1)

There is no universal “qubit operator matrix” interchange standard.
paulikit accepts two dense layouts that share one streaming reader
after a tiny header. Compression codecs are reserved
(`codec: "identity"` only in v1); use sparse or Pauli form when the
operator is compressible by structure.

### Layout A — raw `complex128` + sidecar JSON (primary)

- Blob: row-major IEEE `complex128`, length `dim * dim`, little-endian
  on little-endian hosts (v1 refuses a mismatched `endianness`).
- Sidecar: `H.c128.json` next to `H.c128` (or pass an explicit path).

```json
{
  "format": "paulikit.dense_c128.v1",
  "dim": 8192,
  "dtype": "complex128",
  "layout": "row-major",
  "endianness": "little",
  "n_qubits": 13,
  "codec": "identity",
  "sha256": "optional hex digest of the raw blob"
}
```

Required fields: `format`, `dim` (power of two), `dtype`, `layout`,
`endianness`. Optional: `n_qubits` (must equal $\log_2\mathrm{dim}$),
`sha256`, `codec` (must be `"identity"` if present).

File size must equal `dim * dim * 16`.

### Layout B — NumPy `.npy` (convenience)

Square, C-contiguous (`fortran_order=false`), `complex128` only.
Header carries shape/dtype; no sidecar required. An optional JSON
sidecar may still carry `sha256`.

Do **not** use `.npz` on this path (decode cost; no mmap of the
payload). PKCP remains an **output** format for Pauli chunks, not a
dense input.

## Library API

```python
from paulikit.algorithms.dense_bucketed import DenseBucketedSource
from paulikit.algorithms.fwht import parallel_decompose_arrays

src = DenseBucketedSource.from_dense_file(
    "H.c128",                    # or "H.npy"
    meta="H.c128.json",          # optional if PATH.json exists; omit for .npy
    chunk_size=256,
    spill_dir="/tmp/H.buckets",  # prefer local SSD, not network FS
    max_resident_buckets=2,
)

for x, z, coeff in parallel_decompose_arrays(
    None,
    operator_source=src,
    chunk_size=src.chunk_size,
    executor="thread",
):
    ...
```

Notes:

- `from_dense_file` always builds the **spilled** bucket layout (Pass‑1
  scatter to disk, then LRU gather). It does not load `dim×dim` into
  RAM.
- `chunk_size` on the source and on `parallel_decompose_arrays` must
  match (the drain copies `src.chunk_size` when you omit the argument).
- Low-level escape hatch: `DenseBucketedSource.from_complex128_file`
  if you already know `dim` and optional `data_offset` (used internally
  for `.npy`).
- Validation helpers live in `paulikit.algorithms.dense_input`
  (`resolve_dense_file`, `DenseFileSpec`, `FORMAT_V1`).

Writing a layout-A file from NumPy:

```python
import json
from pathlib import Path
import numpy as np

H = ...  # square complex128, dim = 2**n
path = Path("H.c128")
np.ascontiguousarray(H, dtype=np.complex128).tofile(path)
meta = {
    "format": "paulikit.dense_c128.v1",
    "dim": int(H.shape[0]),
    "dtype": "complex128",
    "layout": "row-major",
    "endianness": "little",
    "n_qubits": int(H.shape[0]).bit_length() - 1,
}
Path(str(path) + ".json").write_text(json.dumps(meta, indent=2))
```

## CLI

```bash
paulikit decompose --operator-file H.c128 --operator-meta H.c128.json \
    --parallel --chunk-size 256 \
    --spill-dir /tmp/H.buckets \
    --executor thread

paulikit decompose --operator-file H.npy \
    --parallel --chunk-size 256 --spill-dir /tmp/H.buckets
```

Requirements and behaviour:

- Requires `--parallel` and `--chunk-size` (same drain as the large-$N$
  sparse demo).
- Cannot be combined with `--stream` (use `--parallel`).
- Skips the synthetic oscillator Hamiltonian (`-n` is ignored).
- Optional: `--max-resident-buckets K` (default 2), `--write-chunks`,
  `--progress`, `--executor`, `--n-workers` — same meanings as on the
  sparse parallel path.
- Default spill directory if omitted: `PATH.buckets` beside the
  operator file.

## Pipeline (unchanged math)

```text
format reader → DenseBucketedSource (spill) → gather_chunk → WHT / drain
```

Bucket $b$ holds rows of the gather tile for
$x \in [b\cdot C,\, (b+1)\cdot C)$, with $C=$ `chunk_size`. Pass‑1
streams matrix rows and scatters $H[p,q]$ into bucket
$\lfloor (p\oplus q)/C\rfloor$. Gather reloads at most
`max_resident_buckets` cold tiles via an LRU. WHT coefficients and
PKCP output are unchanged.

## Performance notes

- Prefer a **local SSD** for `--spill-dir` / `spill_dir` (network file
  systems can dominate runtime).
- Layout A and B have essentially the same payload size; `.npy` adds
  a ~128-byte header. Choose A for a portable contract, B when you
  already have NumPy dumps.
- Do not expect gzip/zstd on unstructured `complex128` tiles to beat
  raw sequential I/O; v1 does not enable them. Nearly sparse operators
  belong on the CSR path instead.
- Pass‑1 currently walks the matrix once per bucket (build cost scales
  with number of buckets × rows). Pick `chunk_size` as you would for
  a resident dense drain; very small chunks increase spill-file count.
  See {ref}`dense-ooc-qa` below for why `chunk_size=2` at large $n$
  is a Pass‑1 issue, not a drain issue, and what is planned next.

(dense-ooc-qa)=
## Q&A

### Which path should I use?

Three paths share the same drain math (`gather_chunk` → WHT →
coefficients) but differ in how the operator is supplied:

| Path | Needs full $H$ in RAM? | Parallel drain? | `chunk_size=2` |
|---|---|---|---|
| Dense fast (`assume_dense=True`) | Yes (~1 / 4 / 16 / 64 GiB at $n=13$–$16$) | Yes (`executor="thread"` / `auto` when kernels are built) | Fine — gather is from a resident array |
| Disk OOC (`from_dense_file` / `--operator-file`) | No (Pass‑1 spill + LRU) | Yes (same drain) | Bad for **Pass‑1 today** (see below); moderate $C$ is fine |
| Sparse / CSR | No dense $H$ | Yes | Fine — publication large-$N$ recipe |

- **Dense fast** — $H$ already fits; skip the sparsity scan; C
  `gather.c` XOR-gathers each chunk from the resident matrix.
- **Disk OOC** — $H$ is the memory wall; keep it as a file. This page.
- **Sparse** — do not densify into `.c128`; use the CSR /
  `--parallel` path instead.

`assume_dense=True` and `--operator-file` are **not** the same knob.
The fast path needs a resident `ndarray`; the disk path builds a
`DenseBucketedSource` and never loads `dim×dim`.

### Is the dense fast path parallel? Compatible with disk I/O?

**Parallel: yes.** `parallel_decompose_arrays(H, assume_dense=True,
executor="thread", …)` partitions $x$-chunks across workers; each
worker calls `DenseResidentSource.gather_chunk` (C `gather.c` when
built) then the WHT / coefficient kernels with the GIL released.

**Disk: no.** That path assumes $H$ is already mapped in RAM. File
input uses `DenseBucketedSource.from_dense_file` instead — same drain
after buckets exist, different Pass‑1 builder. You cannot point
`assume_dense=True` at a file and get the resident C gather.

At $n=15$–$16$ the dense-fast wall is simply **holding** $H$
(16 GiB / 64 GiB). Disk OOC is for when that fails; then Pass‑1 — not
drain parallelism — dominates until the builder below is fixed.

### Why is `chunk_size=2` at $n=15$–$16$ impractical “today”?

“Today” means the **current multi-pass Pass‑1** in
`DenseBucketedSource._build_spilled`: for each bucket it re-reads
every row of $H$ and scatters into that bucket. Number of buckets is
$\lceil\mathrm{dim}/C\rceil$, so I/O scales like
$(\mathrm{dim}/C)\cdot|H|$:

| $n$ | $\|H\|$ | Buckets at $C=2$ | Pass‑1 I/O (order of) |
|---|---|---|---|
| 13 | 1 GiB | 4096 | ~4 TiB |
| 14 | 4 GiB | 8192 | ~32 TiB |
| 15 | 16 GiB | 16384 | ~256 TiB |
| 16 | 64 GiB | 32768 | ~2 PiB |

With a **moderate** $C$ (e.g. 256–4096) the same builder is heavy but
finite — local ladder runs reached $n=13$–$14$ that way. After
buckets exist, the **drain** can still use small chunks; the explosion
is build cost, not WHT tile size. Resident dense / sparse drains with
`chunk_size=2` are unaffected.

### Is there a plan to fix that? How does it relate to the C kernels?

Yes — two layers, algorithm first:

1. **Single-pass Pass‑1** (Python or C). One sequential read of $H$;
   each row scatters into the correct open bucket buffer(s) / spill
   files. Target cost $\sim|H|$ bytes read once, not
   $(\mathrm{dim}/C)\cdot|H|$. That is what makes $C=2$ at large $n$
   realistic on disk. No change to drain math or PKCP.
2. **Optional native scatter kernel** (new C/Cython, `nogil`). A tight
   $p\oplus q \mapsto$ bucket-cell loop so the single pass is not
   NumPy-mask bound — same *kind* of win `gather.c` gave the resident
   fancy-index gather. This is **not** an extension of `gather.c`
   (that kernel still assumes a full resident operator). Formats,
   sidecar validation, LRU policy, and CLI stay in Python.

What already lives in C and is **unchanged** by disk OOC: WHT
butterfly, coefficient extraction, Hermiticity check, Pauli labels,
and resident `gather.c`. Once a spill tile is in RAM, drain already
rides those kernels. The missing native piece is **Pass‑1 scatter**,
and only after (or together with) the single-pass redesign — porting
today’s multi-pass NumPy loop to C would only shave constants on an
impractical I/O shape.

## Related docs

- {doc}`tutorial` — step-by-step
  {ref}`Dense operators on disk (I/O) <dense-operators-on-disk>`:
  generate layout A/B, spill, drain, optional PKCP write/read, plus a
  short path Q&A that points here for the full tables
- {doc}`runtime_estimates` — discard vs write vs materialise (sparse
  planning); dense *input* size is a separate constraint
- {doc}`api/algorithms` — autodoc for `dense_input`, `dense_bucketed`,
  `operator_source`, `fwht`
- {doc}`package_layout` — where the modules live in the tree
