# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, the public API may change between minor
versions; see the README's Status section.

## [Unreleased]

## [0.1.3] - 2026-09-28

### Fixed

- **Author spelling.** Coauthor family name corrected from Nasouri to
  Nasoori in `AUTHORS`, `CITATION.cff`, `pyproject.toml`, and the
  Sphinx author string (`docs/conf.py`). Packaging / citation metadata
  only; no change to algorithms, APIs, or measured behaviour.

### Notes

- Intended as a clean republish of the 0.1.2 feature set under corrected
  author metadata. Prior PyPI releases `0.1.0`–`0.1.2` are to be removed
  or yanked after this version is live (those filenames cannot be
  reused).
- Zenodo software DOI for this version:
  [10.5281/zenodo.23009189](https://doi.org/10.5281/zenodo.23009189)
  (concept [10.5281/zenodo.22992251](https://doi.org/10.5281/zenodo.22992251)).
  Matching measurements package 1.0.1:
  [10.5281/zenodo.23009214](https://doi.org/10.5281/zenodo.23009214).

## [0.1.2] - 2026-09-27

### Added

- **CLI stream writer (`--write-chunks PATH`).** Main-drain-only PKCP
  frames (symplectic `x`, `z`, `coeff`); workers never touch the file.
  Alias `--checkpoint-path`. Without it, large CLI runs still count
  and discard. Library: branchless null / checkpoint chunk sinks on
  the drain path.
- **CLI `--progress`.** Opt-in chunk progress on stderr (`k/N`,
  percent, ETA); quiet drain loop free of per-chunk progress calls.
  Parallel/stream paths pre-count planned chunks before the timed
  drain so quiet and progress share the same warm-up.
- **GNU-style CLI help.** Multi-line usage, wrapped description/epilog,
  preserved author newlines, Examples on every subcommand; accurate
  decompose vs library scope (demo Hamiltonian vs arbitrary operators).
- **Runtime estimates doc.** `docs/runtime_estimates.md` — discard vs
  write vs materialise; modest-HW topology (no CPU SKU); hour-line
  ~$2\times 10^{11}$ terms ≈ $N\sim 1.1\times 10^{3}$ / ~20 qubits for
  this oscillator family.

### Changed

- Tutorial / README / Sphinx links for writer, progress, and runtime
  estimates; meson installs `cli_help.py`.

## [0.1.1] - 2026-09-26

### Changed

- **Serial `pauli_label_native` in manylinux wheels.** The label
  kernel no longer requires oneTBB at build time: serial C + Cython
  ships in the Linux wheels alongside `wht_kernel` and `cache_probe`.
  The optional oneTBB parallel fill moved to
  `pauli_label_parallel_native` (from-source / when TBB is present).
  Measured e2e `list[str]` labeling gains only ~1.03× from TBB because
  Python string construction dominates; wheels therefore do not vendor
  TBB for that path. `-Dnative=enabled` now means the serial kernel is
  required; oneTBB is never required for that option.

## [0.1.0] - 2026-09-26

First public release (PyPI target). Package contents were prepared
2026-09-10; this dated entry is the freeze for the first tagged /
uploaded artifacts.

### Added

- **Pauli decomposition by fast Walsh-Hadamard transform.**
  `paulikit.algorithms.fwht` decomposes any $2^n \times 2^n$ complex matrix
  into its Pauli-basis coefficients in $O(n \cdot 4^n)$. Hermitian and
  non-Hermitian input take the same transform; `assume_hermitian=True`
  (the default) returns real coefficients and verifies the input
  really is Hermitian rather than assuming it.

- **Bounded memory.** The active-x set is chunked and each chunk is
  thresholded and emitted before the next is computed, so peak
  resident memory is set by `chunk_size` rather than by the size of
  the result. Sparse input is kept sparse throughout.

- **Streaming entry points.** `fwht_pauli_terms_iter` and
  `parallel_decompose_arrays` yield per chunk, so a caller can consume
  and discard results that would not fit in memory if accumulated.

- **Multi-core execution.** `parallel_decompose` and
  `parallel_decompose_arrays` distribute independent chunks across a
  process pool or a thread pool. Worker count is taken from the CPUs
  actually available to the process rather than from `os.cpu_count()`,
  so a cgroup- or cpuset-restricted machine is not oversubscribed.
  Threaded drain (`executor="thread"` / `"auto"`) uses physical-core
  pinning, capacity-weighted slices, work-stealing, and optional eager
  thread spin-up.

- **Checkpoint and restart.** A binary chunk-framed checkpoint format
  cheap enough to leave permanently enabled. Frames carry their own
  chunk index, so completed work replays on resume regardless of the
  order a pool returns it in, and a torn tail means exactly "that
  chunk was not recorded".

- **Optional compiled kernels.** Under three meson options:
  `wht_kernel` (Walsh–Hadamard butterfly, fused coefficients, dense
  gather, Hermiticity check, with optional x86-64-v3 twins), `native`
  (Pauli label kernel with a oneTBB-parallel variant), and
  `cache_probe` (empirical cache-latency probe). All are optional: the
  package is correct and installable with no C toolchain, and reports
  once when a fallback path is taken. Manylinux wheels ship
  `wht_kernel` (+ `cache_probe`) with `-Dnative=disabled` (no oneTBB
  in the wheel image); label generation falls back to pure Python.

- **Cache-aware auto-tuning.** `paulikit.algorithms.autotune` sizes
  chunks against measured cache boundaries, interpolating between
  measured anchor points rather than applying a closed-form fit.

- **Command-line interface.** `paulikit decompose`, `paulikit
  benchmark` and `paulikit regenerate-fixtures`.

- **Exhaustive verification.** Every term of a decomposition checked
  individually against an independently derived projection oracle -
  not sampled - up to 91,652,096 terms at 14 qubits, Hermitian and
  non-Hermitian. Artifacts and method under `verification/`.

- **Test suite.** 323 default tests (8 further slow benchmark
  comparisons excluded by default).

- **Documentation.** Installation, tutorial, theory, background,
  non-Hermitian operators, annotated source tree, and an API
  reference built from the docstrings.

- GitHub Actions workflow (`.github/workflows/paulikit-wheels.yml` on
  the GitHub mirror,
  [github.com/beavernets-inc/paulikit](https://github.com/beavernets-inc/paulikit))
  that builds manylinux wheels (x86_64 / aarch64, CPython 3.10–3.13)
  and an sdist on tag / manual dispatch, with an optional Trusted
  Publishing upload job. Codeberg remains the canonical repository;
  hosted Forgejo runners are not used for the manylinux matrix.

### Changed

- README Usage leads with the measured sparse/dense fastest paths;
  Sphinx tutorial, installation, and native docs match the multi-kernel
  surface; `--chunk-size` help no longer claims auto-tuning is absent
  from the library APIs.
- GitHub mirror moved to
  [beavernets-inc/paulikit](https://github.com/beavernets-inc/paulikit);
  install docs lead with `pip install paulikit` for the first PyPI
  upload.

### Fixed

- `make check` no longer dumps expected autotune/fork warnings from
  intentional no-probe and process-pool tests.

### Removed

- Stale Status/CHANGELOG claim of an unexplained 14-to-15 qubit
  parallel-efficiency step under the process-pool drain.

### Known limitations

- Linux manylinux wheels ship `wht_kernel` (+ `cache_probe`) but omit
  `pauli_label_native` (oneTBB). macOS / Windows / musllinux use the
  sdist. Install with `pip install paulikit`.
- CPU pinning and topology detection are Linux-only, with a documented
  fallback elsewhere; the non-Linux paths are not exercised in CI.

[Unreleased]: https://codeberg.org/beavernets/paulikit/compare/v0.1.3...HEAD
[0.1.3]: https://codeberg.org/beavernets/paulikit/releases/tag/v0.1.3
[0.1.2]: https://codeberg.org/beavernets/paulikit/releases/tag/v0.1.2
[0.1.1]: https://codeberg.org/beavernets/paulikit/releases/tag/v0.1.1
[0.1.0]: https://codeberg.org/beavernets/paulikit/releases/tag/v0.1.0
