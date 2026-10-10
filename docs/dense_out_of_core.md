# Dense operators that do not fit in RAM

Peak resident memory during a Pauli drain is usually set by
**chunk size**, not by the number of Pauli terms. A second wall
appears when the **input** operator is dense: a full
$2^n \times 2^n$ `complex128` matrix needs $16 \cdot 4^n$ bytes
(about 4 GiB / 16 GiB / 64 GiB at $n = 14,15,16$). Streaming the
*output* does not help if you first allocate that matrix in RAM.

This page documents the **opt-in** out-of-core dense path: keep $H$
on disk, run a Pass‑1 that memmaps $H$ and writes fixed-height
$x$-spill tiles (bit-indexed C fill when built), then feed the same
`gather_chunk` → WHT → drain pipeline used for resident dense and
sparse work.

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
    chunk_size=2,                # drain / WHT tile (cache + parallelism)
    spill_bucket_rows=1024,      # Pass-1 tile; power of two, multiple of chunk_size
    spill_dir="/tmp/H.buckets",  # prefer fast local volume (Pass-1 is write-bound)
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
  write to disk, then slice gather). It does not load `dim×dim` into
  RAM as a dense `ndarray` (it memmaps $H$ for Pass‑1 only).
- **`chunk_size`** ($C$) is the drain tile. **`spill_bucket_rows`**
  ($R$, default: same as `chunk_size`) is the on-disk Pass‑1 tile
  height: must be a **power of two** and a multiple of `chunk_size`
  (so `chunk_size` is also a power of two when spilling). Prefer a
  **small** drain $C$ (e.g. 2) with a **larger** $R$ (e.g. 256–4096)
  so drain keeps thin WHT tiles while Pass‑1 writes fewer, taller
  spill files.
- Pass‑1 bit-routes each cell as $q = p \oplus (x_{\mathrm{lo}}+r)$
  in `pass1_scatter_native` when built (OpenMP over $p$), then writes
  each tile with large POSIX `write()` chunks. Without the extension,
  a NumPy bit-route fallback is used. Across all tiles, each $H[p,q]$
  is read **once**.
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
    --parallel --chunk-size 2 --spill-bucket-rows 1024 \
    --spill-dir /tmp/H.buckets \
    --executor thread

paulikit decompose --operator-file H.npy \
    --parallel --chunk-size 2 --spill-bucket-rows 1024 \
    --spill-dir /tmp/H.buckets
```

Requirements and behaviour:

- Requires `--parallel` and `--chunk-size` (same drain as the large-$N$
  sparse demo).
- Cannot be combined with `--stream` (use `--parallel`).
- Skips the synthetic oscillator Hamiltonian (`-n` is ignored).
- Optional: `--spill-bucket-rows R` (power of two and multiple of
  `--chunk-size`; default = chunk size), `--max-resident-buckets K`
  (default 2), `--write-chunks`, `--progress`, `--executor`,
  `--n-workers`.
- Default spill directory if omitted: `PATH.buckets` beside the
  operator file.

## Pipeline

```text
memmap H → Pass-1 (fill tile R × dim, write spill file) → …
  … gather_chunk (thin C-row slice) → WHT / coeffs / drain
```

Geometry (spilled path):

- Spill tile $b$ covers gather-rows
  $x \in [b\cdot R,\, b\cdot R + \mathrm{height})$ with
  $R=$ `spill_bucket_rows`.
- Drain requests aligned chunks of height $C=$ `chunk_size` with
  $R$ a multiple of $C$, so a drain chunk never crosses a spill-file
  boundary.
- Pass‑1 fills one spill tile at a time (peak ≈ one $R\times\dim$
  tile plus the memmap of $H$). Gather reloads at most
  `max_resident_buckets` drain slices via an LRU.

WHT coefficients and PKCP output are unchanged from the resident path.

## Performance notes

- Prefer a **fast local volume** for `--spill-dir` / `spill_dir`. After
  the bit-indexed C fill, Pass‑1 is almost entirely spill **writes**
  (one full copy of $|H|$ to disk). USB enclosures and network FS often
  dominate wall-clock even when the media is SSD.
- Spill tiles use large POSIX `write()` chunks
  (`pass1_scatter_native.write_c128_file` when built).
- Order-of-magnitude (measured): at $n=14$ ($\lvert H\rvert=4$ GiB),
  $C=2$, $R=1024$, 4 drain workers, spill on an internal SATA SSD —
  Pass‑1 ≈ 10 s (fill ≈ 2 s, write ≈ 9 s) and drain ≈ 2.5 s. The same
  Pass‑1 with spill on a slower USB SSD volume is write-bound much
  longer. Treat these as orientation, not a published SLA.
- Layout A and B have essentially the same payload size; `.npy` adds
  a ~128-byte header. Choose A for a portable contract, B when you
  already have NumPy dumps.
- Do not expect gzip/zstd on unstructured `complex128` tiles to beat
  raw sequential I/O; v1 does not enable them. Nearly sparse operators
  belong on the CSR path instead.

(dense-ooc-qa)=
## Q&A

### Which path should I use?

Three paths share the same drain math (`gather_chunk` → WHT →
coefficients) but differ in how the operator is supplied:

| Path | Needs full $H$ in RAM? | Parallel drain? | `chunk_size=2` |
|---|---|---|---|
| Dense fast (`assume_dense=True`) | Yes (~1 / 4 / 16 / 64 GiB at $n=13$–$16$) | Yes (`executor="thread"` / `auto` when kernels are built) | Fine — gather is from a resident array |
| Disk OOC (`from_dense_file` / `--operator-file`) | No (Pass‑1 spill + slice gather) | Yes (same drain) | Use with **large** power-of-two `spill_bucket_rows`; do not set spill height to 2 |
| Sparse / CSR | No dense $H$ | Yes | Fine — publication large-$N$ recipe |

- **Dense fast** — $H$ already fits; skip the sparsity scan; C
  `gather.c` XOR-gathers each chunk from the resident matrix.
- **Disk OOC** — $H$ is the memory wall; keep it as a file. This page.
- **Sparse** — do not densify into `.c128`; use the CSR /
  `--parallel` path instead.

`assume_dense=True` and `--operator-file` are **not** the same knob.
The fast path needs a resident `ndarray`; the disk path builds a
`DenseBucketedSource` and never loads `dim×dim` as a dense array.

### Is the dense fast path parallel? Compatible with disk I/O?

**Parallel: yes.** `parallel_decompose_arrays(H, assume_dense=True,
executor="thread", …)` partitions $x$-chunks across workers; each
worker calls `DenseResidentSource.gather_chunk` (C `gather.c` when
built) then the WHT / coefficient kernels with the GIL released.

**Disk: no.** That path assumes $H$ is already an in-RAM array. File
input uses `DenseBucketedSource.from_dense_file` instead — same drain
after spill tiles exist, different Pass‑1 builder. You cannot point
`assume_dense=True` at a file and get the resident C gather.

At $n=15$–$16$ the dense-fast wall is simply **holding** $H$
(16 GiB / 64 GiB). Disk OOC is for when that fails; then Pass‑1
**writes** (not drain parallelism) usually dominate wall-clock.

### Why was `chunk_size=2` on disk impractical, and what changed?

Historically, Pass‑1 tile height was tied to drain $C$. Then $C=2$
meant $\lceil\mathrm{dim}/2\rceil$ spill tiles and that many full
re-reads of $H$ in a Python/NumPy scatter (hundreds of TiB of H I/O
at $n=15$). Thin drain tiles were coupled to tiny Pass‑1 tiles.

**Now:**

1. **Decouple** drain $C$ from spill height $R=$ `spill_bucket_rows`
   ($R$ power of two, multiple of $C$).
2. **Memmap** $H$ and bit-route $q = p \oplus (x_{\mathrm{lo}}+r)$ in
   C so each operator cell is read once across all tiles.
3. **Write** tall spill files with large POSIX `write()` chunks;
   `gather_chunk` reads only the thin drain slice.

So $C=2$ on disk is a supported recipe when $R$ is large and
`--spill-dir` is on a fast volume. Remaining Pass‑1 cost is mostly
writing $|H|$ bytes once.

### How does this relate to the C kernels?

| Stage | Kernel (when built) |
|---|---|
| Resident dense gather | `gather.c` / `gather_native` |
| Disk Pass‑1 fill + spill write | `pass1_scatter_native` (sibling of `gather`, not an extension of it) |
| Drain | WHT / coeffs / Hermiticity (unchanged) |

If `pass1_scatter_native` is absent, Pass‑1 falls back to a NumPy
bit-route and `ndarray.tofile` with identical results.

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
