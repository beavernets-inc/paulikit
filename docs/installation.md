# Installation

Full installation reference. For the common case, the README's
Installation section is enough; this document covers the build
system's requirements, the editable-install sequencing, and the
optional native extensions in detail.

## From PyPI

```bash
pip install paulikit
```

Linux manylinux wheels (CPython 3.10–3.13, x86_64 and aarch64) ship
`wht_kernel`, `cache_probe`, and serial `pauli_label_native` (C +
Cython). The optional oneTBB parallel label module is omitted from
wheels on purpose. Other platforms install from the sdist and compile
what the local toolchain allows, or fall back to pure Python.

## From a source checkout

paulikit is built with [meson-python](https://mesonbuild.com/meson-python/)
(the same build backend NumPy and SciPy use), and optionally compiles
several Cython/C(/C++) kernels that accelerate the transform, gather,
coefficient emission, Hermiticity check, Pauli labels, and cache
probe — see [Native extensions](#native-extensions) below.

**Recommended for development:** run `./configure` from this
directory. It creates/reuses a dedicated venv (default
`~/.venvs/paulikit`, deliberately outside the source tree — meson
rejects an absolute in-tree numpy include path if the venv lives
inside the project root), prints an itemized capability/environment
diagnostic report (compiler, Cython, TBB, cache hierarchy, NumPy's
BLAS backend, etc.; run `./configure --help` for the full list), and
generates a `Makefile` with the standard GNU set of
targets (`all`/`build`/`install`/`install-strip`/`installdirs`/
`check`/`test`/`installcheck`/`uninstall`/`docs`/`dist`/`TAGS`/
`clean`/`mostlyclean`/`distclean`/`maintainer-clean`/`report`) that
handle the `--no-build-isolation` editable-install sequencing below
correctly and automatically:

```bash
./configure                 # accepts --prefix/--docdir/--srcdir/VAR=value
                             # and the standard GNU dir-var options too, see --help
make                         # = make all = make build (editable install)
make check                   # run the test suite
```

If you'd rather not use `./configure`, the manual sequence it
automates is:

```bash
pip install numpy meson-python cython ninja
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required for editable installs: without it,
NumPy's include path gets baked in from a throwaway build-isolation
environment that goes stale on later rebuilds (this is NumPy's own
documented practice for meson-python editable installs, not a
paulikit-specific quirk) — and `numpy`/`meson-python`/`cython`/`ninja`
must already be installed in the target environment first, since
`--no-build-isolation` means pip won't fetch them into a throwaway
env for you the way it normally would. A regular, non-editable
`pip install .` does not need any of this.

With test/profiling dependencies:

```bash
pip install -e ".[test]" --no-build-isolation   # pytest, scipy (what CI installs)
pip install -e ".[reference]" --no-build-isolation  # PennyLane, for fixture regeneration and the PennyLane comparisons
pip install -e ".[dev]" --no-build-isolation    # test and reference, plus Cython and ruff
pip install -e ".[docs]" --no-build-isolation   # Sphinx / Furo / MyST stack for `make docs`
pip install -e ".[sparse]" --no-build-isolation # just scipy, for build_hamiltonian(sparse=True) / pad_to_power_of_two(sparse=True)
```

Requires Python >= 3.10. Runtime dependencies are just `numpy` — the
core algorithms have no dependency on PennyLane, Qiskit, or Classiq;
those are only used in the `reference`/`dev` extras, for generating and
cross-checking correctness fixtures. `scipy` is likewise optional:
only needed for the `sparse=True` path on `build_hamiltonian`,
`pad_to_power_of_two`, and (as an input type)
`fwht_pauli_coefficients`/`fwht_pauli_terms` -
calling any of those with `sparse=True` (or passing a
`scipy.sparse` operator directly) without `scipy` installed raises a
clear `ImportError` naming the `sparse` extra, rather than silently
falling back to the dense path. Building from source always
requires `meson-python`, `Cython`, and `numpy` (PEP 517/518's
`[build-system] requires` has no conditional mechanism), but both are
pure-Python-installable — no C toolchain is needed just to build the
pure-Python parts of the package.


## Native extensions

Three independent meson feature options gate the compiled modules
(defaults are all `auto`: build if the toolchain is present, otherwise
fall back quietly at import time with a one-time warning when a
fallback path first runs):

| meson option | Modules | Needs |
|---|---|---|
| `wht_kernel` | `wht_native` (+ optional `wht_native_v3`), `coeffs_native` (+ optional `coeffs_native_v3`), `gather_native`, `hermitian_check_native` | C compiler + Cython ≥ 3.0 |
| `native` | `pauli_label_native` (serial); optional `pauli_label_parallel_native` | Serial: C + Cython ≥ 3.0. Parallel fill: C++ + [oneTBB](https://github.com/oneapi-src/oneTBB) |
| `cache_probe` | `cache_probe` | C compiler + Cython ≥ 3.0 |

The `wht_kernel` set carries the transform, gather, coefficient
emission, and Hermiticity check — these are the modules that matter
for throughput and for the threaded drain. Serial `pauli_label_native`
accelerates label-string generation for the dict-returning APIs and
ships in the Linux wheels. The optional `pauli_label_parallel_native`
module needs oneTBB and is from-source only (`parallel_labels=True`);
e2e it barely helps because Python `str` construction dominates.
`cache_probe` feeds `paulikit.algorithms.autotune`'s chunk sizing.

The optional `_v3` twins of the WHT and coefficient kernels are the
same source compiled a second time under `-march=x86-64-v3` when the
build toolchain accepts that flag. The public API picks between a
module and its `_v3` twin at **import** time from a runtime CPU check,
never at build time, so a wheel built on a newer CPU stays correct
(just slower) on an older one.

To force behaviour explicitly:

```bash
# Fail the build if a feature cannot be compiled:
pip install -e . --no-build-isolation \
  --config-settings=setup-args="-Dwht_kernel=enabled" \
  --config-settings=setup-args="-Dnative=enabled" \
  --config-settings=setup-args="-Dcache_probe=enabled"

# Force pure-Python / declared-cache fallbacks:
pip install -e . --no-build-isolation \
  --config-settings=setup-args="-Dwht_kernel=disabled" \
  --config-settings=setup-args="-Dnative=disabled" \
  --config-settings=setup-args="-Dcache_probe=disabled"
```

On a from-source install the compiled extensions remain optional
accelerators: requiring a C toolchain for every `pip install` of the
sdist would be too heavy a default. Published Linux wheels already
include `wht_kernel`, `cache_probe`, and serial `pauli_label_native`.
Wheel CI runs on tag / manual dispatch via
`.github/workflows/paulikit-wheels.yml` on the GitHub mirror
([github.com/beavernets-inc/paulikit](https://github.com/beavernets-inc/paulikit));
Codeberg remains the canonical tree. Making every kernel a hard
sdist requirement (NumPy/SciPy model) is still a later packaging step;
oneTBB is not part of that gate for serial labels.
