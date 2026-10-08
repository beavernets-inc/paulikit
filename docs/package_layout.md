# Package layout

```
src/paulikit/
    __init__.py           Package metadata, public API summary.
    hamiltonian.py         Coupled-oscillator Hamiltonian construction
                            (NumPy implementation, with an optional
                            scipy.sparse output path).
    pauli_utils.py          Minimal, dependency-free Pauli-matrix
                            helpers (label <-> matrix, reconstruction).
    algorithms/
        __init__.py
        fwht.py             Fast Walsh-Hadamard Transform decomposition
                            (the shipped algorithm).
        autotune.py          Runtime auto-tuning for chunk size
                            (against the machine's measured cache
                            hierarchy) and the streaming-vs-dense
                            decision (against available memory).
        operator_source.py   `gather_chunk` adapters (resident dense
                            XOR gather, sparse nnz scatter).
        dense_input.py       On-disk dense layouts A/B: validate raw
                            `complex128` + sidecar JSON or `.npy`.
        dense_bucketed.py    Opt-in spilled x-buckets and
                            `from_dense_file` factory for out-of-core
                            dense input (see dense_out_of_core.md).
    testing/
        __init__.py
        fixtures.py         Known-good Hamiltonians and their
                            independently-verified expected Pauli
                            decompositions, for use by the algorithm's
                            tests.
    _native/                Optional compiled extensions used by
                            `algorithms/fwht.py` and `autotune.py`
                            when available, each with a pure-Python or
                            NumPy fallback otherwise. See
                            installation.md. Gated by three
                            `meson.options` features:

                            - `wht_kernel`: `wht_native` /
                              `wht_native_v3`, `coeffs_native` /
                              `coeffs_native_v3`, `gather_native`,
                              `hermitian_check_native`
                            - `native`: serial `pauli_label_native`
                              (Cython/C); optional
                              `pauli_label_parallel_native`
                              (Cython/C++ + oneTBB)
                            - `cache_probe`: `cache_probe`
    cli.py                  Command-line interface wiring the above
                            together into subcommands.
    meson.build             Per-directory Meson build rules (one per
                            subpackage above, plus a top-level
                            `meson.build` and `meson.options` at the
                            repository root of this package).
tests/
    test_fixtures.py        Self-consistency checks for the fixtures.
    test_fwht.py             Correctness tests for algorithms/fwht.py.
    (and further test modules covering autotuning, checkpointing,
    streaming, parallel decomposition, sparse input, dense bucketed /
    file layouts, and the native kernels — see the `tests/` directory
    for the full list).
```

`hamiltonian.py` and `pauli_utils.py` sit at the package root (not
under `algorithms/`) because they are shared infrastructure for the
decomposition pipeline, not algorithm-specific internals.
