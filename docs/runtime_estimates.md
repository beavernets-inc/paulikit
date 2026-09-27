# Runtime estimates (when does a run take hours?)

This note is for planning large **sparse oscillator** decompositions
on modest hardware. Numbers below are **order-of-magnitude guidance**,
anchored to published Zenodo measurements and local CLI checks on the
same machine class — not a promise for every CPU, disk, or governor.

The CLI `paulikit decompose` always builds the synthetic coupled-oscillator
Hamiltonian. The same timing picture applies when you call
`parallel_decompose_arrays` (or the streaming iterators) on a similar
sparse workload from Python.

## Reference hardware (modest / low-end class)

Wall times in this note come from a **laptop-class** box used for the
Zenodo publication benches and CLI checks. Useful for planning; not a
SKU endorsement.

| Resource | Typical on that box |
|---|---|
| Physical cores | 4 (8 logical with SMT) |
| L1d / L1i | 128 KiB each (4 instances) |
| L2 | 1 MiB (4 instances) |
| L3 | 8 MiB (shared) |
| System RAM | ~16 GiB |
| Governor / clocks | Often `powersave`; base ~1.8 GHz with turbo up to ~4 GHz — short runs swing with DVFS |

**What we omit on purpose:** the marketing CPU model string. Topology
and memory matter for “will this fit / how many workers?”; the exact
SKU does not change the discard-vs-write-vs-materialise story. The
Zenodo result JSON already records `cpu_model` (and topology) for
bit-for-bit reproduction if you need it.

On this class of machine, sparse N=300 discard peaks around
**~140–150 MiB** RSS; dense qubits=13 benches are multi-GiB because of
the dense input, not because streaming fails.

## Three modes (do not mix them up)

| Mode | What happens | Memory / disk | Typical wall (this class of HW) |
|---|---|---|---|
| **Discard** | Drain chunks; count terms; drop arrays | Peak RSS tracks chunk work (~tens–low hundreds of MiB) | N=150: ~sub-second–~1 s; N=300: **~10–25 s** |
| **Checkpoint / write** | Same compute, plus PKCP frames via `--write-chunks` / `checkpoint_path` | Same peak RSS **plus** archive size ≈ O($n_{\mathrm{terms}}$) on disk | Discard time + I/O (SSD: small; slow disk: can dominate) |
| **Materialise** | Build one giant `dict[str, complex]` (or equivalent) for all terms | Grows with **every** term; not the streaming path | **Avoid** at $N\gtrsim 150$ — hours and/or OOM long before discard would finish |

Publication and `perf` benches use **discard** (quiet drain). Prefer
that for timing claims. Use write when you need a usable on-disk
result. Do not materialise the full label dict at these sizes.

## Anchors (published + CLI)

Prefer **cycle counts** from Zenodo for paper ratios; wall times here
are for **planning**.

**Sparse N=300, discard, single-core**  
(`publication_single_core_20260925_113832.json`, paulikit `811d4b5`):

- Terms: $1\,470\,038\,016$
- `decompose_s` mean ≈ **24.6 s** (full-call wall ≈ 25.6 s including build)

**Sparse N=300, discard, multi-core**  
(`publication_multi_core_auto_20260920_122452.json`, and local CLI):

- Zenodo `auto`, `n_workers=4`: decompose ≈ **12.7 s**
- Zenodo `auto`, `n_workers=8`: decompose ≈ **17.8 s** (SMT siblings; often no win)
- Local CLI on the same box
  (`--parallel --executor thread --n-workers 4 --chunk-size 2 --progress`):
  ≈ **10.7 s** for the same $1\,470\,038\,016$ terms

So N=300 discard on this class is roughly **~10–25 s** depending on
executor, worker count, warmth, and DVFS — not a single fixed number.

**Sparse N=150, discard** (same family; CLI / paper scale):

- Terms: $91\,652\,096$
- Parallel drain: about **0.5–1 s** on a warm machine (varies with DVFS)

**Sparse N=150, write (PKCP)** (local CLI check on the same class of HW):

- Archive on the order of **~1–2 GiB**
- Extra wall vs discard on a fast SSD: about **~1 s** — I/O is cheap
  relative to compute at this size; on a slow HDD or network FS it can
  grow to minutes while compute stays the same

## Rough cost model (discard)

Treat discard wall as roughly proportional to nonzero term count for
this Hamiltonian family and fixed `chunk_size` (e.g. 2):

$$
T_{\mathrm{discard}} \approx \alpha \cdot n_{\mathrm{terms}}
$$

From N=300 **single-core** (`decompose_s` ≈ 24.6 s,
$n_{\mathrm{terms}} \approx 1.47\times 10^{9}$):

$$
\alpha_{\mathrm{1\,core}} \approx 1.7\times 10^{-8}\,\mathrm{s/term}
$$

From the same N=300 on **4 workers / thread** (~10.7 s):

$$
\alpha_{\mathrm{4\,workers}} \approx 7\times 10^{-9}\,\mathrm{s/term}
$$

**Hour-scale (≈ 3600 s) on the single-core α:**

$$
n_{\mathrm{terms}} \sim \frac{3600}{\alpha_{\mathrm{1\,core}}} \approx 2\times 10^{11}
$$

That is roughly **~140×** the N=300 term count ($1.47\times 10^{9}$).

**What $N$ / qubit count is that?** For this fully-coupled oscillator
family, unpadded dimension is $\dim = N + N(N+1)/2$, then
$n_{\mathrm{qubits}}=\lceil\log_2\dim\rceil$ after padding. Surviving
term counts from the Zenodo sparse sweep ($N=50\ldots300$) fit roughly

$$
n_{\mathrm{terms}} \sim N^{3.9}
$$

(order-of-magnitude only; residuals ~10–30% on the measured points).
Inverting to $n_{\mathrm{terms}}\approx 2\times 10^{11}$ gives about

| | Estimate |
|---|---|
| Oscillators $N$ | **~$\mathbf{1.1\times 10^{3}}$** (ballpark $10^{3}$–$1.2\times 10^{3}$) |
| Padded qubits | **~$\mathbf{20}$** ($\dim\sim 6\times 10^{5}$, next power of two $2^{20}$) |

So on this machine class, an hour of **discard** compute is not N=300
(tens of seconds) — it is a problem about **four× larger in $N$**, two
qubit-doublings past the N=300 / 16-qubit anchor. Hours appear earlier
if you leave discard/write streaming (materialise), or if the disk is
slow on a huge PKCP write.

On the **same** α, N=300 stays tens of seconds; the table above is the
single-core hour-line for this Hamiltonian family only.

**Still-slower hardware** (fewer than 4 cores, colder turbo, heavy
background load): treat α as **2–5× worse** than the single-core
figure. Then N=300 discard may be **1–2 minutes**, and the “one hour”
term count drops by that same factor. Always re-check on your box
(below).

**Write path:** add time ≈ `(archive bytes) / disk write bandwidth`.
Archive size scales roughly with $n_{\mathrm{terms}}$. Example: if
N=150 is ~1.5 GiB, N=300 (~16× terms) is tens of GiB — still usually
minutes on SSD, can approach an hour on slow media while discard
compute remains ~tens of seconds.

**Materialise:** cost is not $\alpha \cdot n_{\mathrm{terms}}$ for the
FWHT alone — it is also allocating and hashing/storing every labelled
term. Plan on **failure** (OOM) or **hours** well before the discard
hour-line.

## Sanity-check on your machine

1. Run a known size with progress, discard only:

   ```bash
   paulikit decompose -n 150 --parallel --chunk-size 2 --progress
   ```

   For the N=300 anchor on a 4-core box:

   ```bash
   paulikit decompose -n 300 --parallel --executor thread \
       --n-workers 4 --chunk-size 2 --progress
   ```

2. Note wall time and term count from the summary line.

3. Extrapolate for discard on the same executor / `chunk_size`:

   $$
   T(N') \approx T(N)\cdot
   \frac{n_{\mathrm{terms}}(N')}{n_{\mathrm{terms}}(N)}
   $$

   For write, add a disk estimate from a small `--write-chunks` trial
   (archive size / observed write rate).

4. If early `--progress` ETA already points at hours, stop and switch
   to discard-only or a smaller $N$ before filling the disk.

## What this note is not

- Not a substitute for the Zenodo measurement deposit or manuscript
  cycle-count claims.
- Not calibrated for dense random Hermitians (different constant).
- Not a guarantee under thermal throttling, busy systems, or
  `--executor process` (often several× slower than `thread`/`auto`
  when kernels are available).

For measurement methodology and raw JSON, see the companion
measurements / Zenodo package (`docs/PUBLICATION_SINGLE_CORE.md`,
`docs/PUBLICATION_MULTI_CORE.md`).
