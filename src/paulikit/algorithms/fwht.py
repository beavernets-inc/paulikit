# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Original Pauli decomposition via the Fast Walsh-Hadamard Transform (FWHT).

Implements the O(N^2 log N) algorithm (N = 2**n) described in
"Pauli decomposition via the fast Walsh-Hadamard transform"
(https://iopscience.iop.org/article/10.1088/1367-2630/adb44d), as an
alternative to an O(4^n)-with-symbolic-overhead brute-force approach.

This is an original implementation: the algorithm's three steps were
independently re-derived and verified against a from-scratch,
definition-level brute-force decomposition (Frobenius inner product
against every tensor-product Pauli string) before writing the fast
version, rather than transcribed from the paper. See
``tests/test_fwht.py`` (package root) for that verification.

See the full documentation site (``docs/``) for a much more detailed
treatment: :doc:`/background` (the physical problem this solves),
:doc:`/theory` (this module's derivation, worked step by step with a
hand-verified example), and :doc:`/non_hermitian` (physical examples
of non-Hermitian operators and a worked complex-coefficient example).

Mathematical basis
-------------------
Using the symplectic (X/Z) representation of an n-qubit Pauli string,
indexed by bitmasks x, z in [0, 2**n):

    P(x, z) = bigotimes_{j=0}^{n-1} i**(x_j & z_j) * X**x_j * Z**z_j

(qubit j corresponds to bit (n-1-j) of x and z, matching the row/column
order produced by a left-to-right ``numpy.kron`` chain).

The matrix element ``<p| X**x Z**z |q>`` equals
``(-1)**popcount(q & z)`` when ``p == q ^ x``, and 0 otherwise. So the
Frobenius inner product coefficient is:

    c(x, z) = (1 / dim) * conj(i**popcount(x & z))
              * sum_q H[q ^ x, q] * (-1)**popcount(q & z)

For fixed x, the inner sum over q, as a function of z, is exactly the
(unnormalized, +-1 butterfly) Walsh-Hadamard Transform of the
"anti-diagonal gather" g_x(q) = H[q ^ x, q]. This is the paper's three
steps: (1) the XOR-index gather/permutation building g_x for every x,
(2) the Walsh-Hadamard Transform applied to each g_x, (3) the
phase-factor multiplication by conj(i**popcount(x & z)) / dim.

Complexity: building all 2**n gathers and running a length-2**n WHT on
each costs O(2**n * n * 2**n) = O(n * 4**n) -- i.e. O(N^2 log N) for
N = 2**n -- versus a naive approach's O(4**n) *symbolic* trace
evaluations, each of which itself costs O(2**n) or worse with symbolic
(e.g. SymPy) overhead. The fast approach also parallelizes trivially
across x (each row's gather + WHT is independent); this module
implements that parallelization (see ``parallel_decompose`` and
``parallel_decompose_arrays`` below).

Hermitian and non-Hermitian operators
---------------------------------------
Nothing in the derivation above requires H to be Hermitian: the Pauli
strings span the full space of 2**n x 2**n complex matrices, so
``fwht_pauli_coefficients`` decomposes *any* complex matrix exactly,
producing complex coefficients in general (real coefficients are the
special case that results from Hermitian input). ``fwht_pauli_terms``
defaults to ``assume_hermitian=True`` for convenience with this
package's primary use case (real-symmetric coupled-oscillator
Hamiltonians) but supports ``assume_hermitian=False`` for arbitrary
operators - relevant for, e.g., non-Hermitian effective Hamiltonians
of open/dissipative systems, PT-symmetric Hamiltonians, Liouvillian
superoperators, or individual non-Hermitian summands of a Hermitian
total.
"""

from __future__ import annotations

import os
import struct
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    from paulikit._native import pauli_label_native as _native
except ImportError:
    _native = None

try:
    from paulikit._native import pauli_label_parallel_native as _native_parallel
except ImportError:
    _native_parallel = None

def _cpu_has_x86_64_v3() -> bool:
    """Best-effort check for the x86-64-v3 feature set (AVX2, BMI2,
    FMA, POPCNT - what ``-march=x86-64-v3`` requires the CPU to
    support), used to pick between the baseline and ``_v3`` compiled
    kernel variants below.

    Reads ``/proc/cpuinfo``'s ``flags`` line. This is Linux/glibc-only,
    matching the existing bias of this package's native layer (see
    ``wht.c``'s ``sysconf(_SC_LEVEL1_DCACHE_SIZE)``, also a glibc
    extension with a silent, safe fallback elsewhere). Returns False -
    the safe choice, since it only ever skips an optimization, never
    causes a wrong dispatch - on any platform or parse failure.

    TODO(portability): this has no answer on non-Linux platforms (no
    ``/proc/cpuinfo`` at all) or on non-glibc Linux where the format
    may differ; it silently falls back to the baseline kernel there
    rather than actually detecting the CPU. A CPUID-based C probe
    (portable, no ``/proc`` dependency) would close that gap - left
    for whoever picks this up next, since it needs its own new
    optional/fallback compiled extension and this project already has
    a working, simpler path for the Linux case that matters today.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return {"avx2", "bmi2", "fma", "popcnt"} <= flags
    except OSError:
        pass
    return False


_HAS_X86_64_V3 = _cpu_has_x86_64_v3()

# Three-tier fallback, resolved once at import time: the AVX2/POPCNT-
# enabled ``_v3`` variant (same C source as the baseline, compiled a
# second time with ``-march=x86-64-v3``; see meson.build) when the CPU
# supports it and the variant was built, else the baseline compiled
# kernel, else pure Python. Resolved once per process and never
# revisited - every thread/process worker uses whichever module-level
# function reference is bound here, so this cannot affect scheduling
# or parallelism, only which compiled code a call reaches.
_wht_native = None
if _HAS_X86_64_V3:
    try:
        from paulikit._native import wht_native_v3 as _wht_native
    except ImportError:
        pass
if _wht_native is None:
    try:
        from paulikit._native import wht_native as _wht_native
    except ImportError:
        _wht_native = None

# Module-level, per-process cache for the WHT kernel's L1-cache tile
# size - the same pattern (and same rationale) as
# ``autotune._cached_l2_bytes``: passing ``tile=0`` makes
# ``paulikit_wht_tile_for_cache`` (wht.c) re-derive the tile from this
# machine's L1 data-cache size via ``sysconf`` on every call, which is
# invariant for the process's whole lifetime (dim-dependent, but not
# time-dependent - the same dim always yields the same tile on the
# same machine). Measured cost of the redundant re-derivation itself
# turned out small on this particular machine (~1.2e5 instructions
# across 16 chunks - `sysconf` here is apparently served from a cached
# glibc value, not a real syscall each time), so this is a cleanliness
# fix rather than a large measured win locally - but it matters more
# than a purely local number suggests: on an HPC node (or any machine
# where `sysconf` genuinely reaches the kernel, or where per-chunk
# call counts are far higher than this investigation's 16), paying a
# redundant syscall-class cost once per chunk instead of once per
# process is exactly the kind of assumption that stops holding at
# scale. Keyed by ``dim`` (not a single scalar) since callers may
# process operators of different sizes within one process lifetime.
# Thread-safe by the same double-checked-locking pattern
# ``autotune._l2_bytes_or_none`` uses: population is lock-guarded, so
# concurrent callers from multiple threads (the thread-executor drain
# path) cannot race to invoke the tile computation more than once per
# dim. Does not protect against separate *processes* racing - each
# has its own independent cache, so there is nothing to race there.
_wht_tile_cache_lock = threading.Lock()
_wht_tile_cache: dict[int, int] = {}


def _wht_tile_for_dim(dim: int) -> int:
    """Cached wrapper around ``wht_native.tile_for_cache(dim)`` - see
    the module-level cache comment above for the full rationale.
    Returns 0 (the kernel's own "derive it yourself" sentinel) if the
    compiled kernel is unavailable, so callers can pass this straight
    through to ``wht_rows_inplace`` unconditionally."""
    if _wht_native is None:
        return 0
    if dim in _wht_tile_cache:
        return _wht_tile_cache[dim]
    with _wht_tile_cache_lock:
        if dim in _wht_tile_cache:
            return _wht_tile_cache[dim]
        tile = int(_wht_native.tile_for_cache(dim))
        _wht_tile_cache[dim] = tile
        return tile

_coeffs_native = None
if _HAS_X86_64_V3:
    try:
        from paulikit._native import coeffs_native_v3 as _coeffs_native
    except ImportError:
        pass
if _coeffs_native is None:
    try:
        from paulikit._native import coeffs_native as _coeffs_native
    except ImportError:
        _coeffs_native = None

try:
    from paulikit._native import gather_native as _gather_native
except ImportError:
    _gather_native = None

try:
    from paulikit._native import hermitian_check_native as _hermitian_check_native
except ImportError:
    _hermitian_check_native = None

# scipy.sparse is imported LAZILY (see _scipy_sparse_module below),
# not at module load time like the extensions above. Measured
# directly (perf stat instructions:u): `import scipy.sparse` alone
# costs ~456M instructions on top of bare NumPy - more than the
# entire remaining decomposition costs on real dense input, and paid
# by every caller of this module regardless of whether they ever pass
# sparse input, since it previously ran unconditionally at import
# time. pauli_lcu, by contrast, never imports scipy at all (measured:
# its own import cost is indistinguishable from bare NumPy's). This
# was the dominant, previously-uncounted cost behind the whole
# dense-vs-pauli_lcu comparison in this investigation - every earlier
# measurement in the performance-investigation plan/PLAN.md already
# paid this cost once per subprocess, in EVERY condition (paulikit
# baseline included), so it did not show up as a per-call regression,
# only as a large constant both this module's own baseline and its
# full runs shared - which is exactly why comparing total instruction
# counts end-to-end (rather than isolating import cost first) missed
# it for this entire investigation until checked directly.
_sp = None
_sp_import_attempted = False


def _scipy_sparse_module():
    """Returns the ``scipy.sparse`` module, importing it on first call
    and caching the result (including the "not installed" case, so a
    missing scipy is retried at most once, not on every call)."""
    global _sp, _sp_import_attempted
    if not _sp_import_attempted:
        _sp_import_attempted = True
        try:
            import scipy.sparse as sp_module
            _sp = sp_module
        except ImportError:
            _sp = None
    return _sp


_POPCOUNT_BYTE_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# The four values i**k can take, indexed by k & 3. The phase factor
# is *always* one of these, so ``1j ** popcount(...)`` - a full
# complex power, evaluated per term - computes a constant the hard
# way. A 4-entry gather replaces it.
_PHASE_BY_POPCOUNT_MOD4 = np.array([1 + 0j, 1j, -1 + 0j, -1j], dtype=complex)

# The conjugates of the above, which is what the coefficient formula
# actually multiplies by (see the module docstring: the factor is
# ``conj(i**popcount(x & z)) / dim``). Conjugating a fourth root of
# unity is an index reflection, ``(4 - k) & 3``, so the conjugated
# table is free to precompute and removes a full-array ``np.conj``
# temporary from the per-chunk path.
_CONJ_PHASE_BY_POPCOUNT_MOD4 = _PHASE_BY_POPCOUNT_MOD4.conj()


def _popcount_array(values: NDArray[np.integer], n_bits: int) -> NDArray[np.integer]:
    """Vectorized population count for integers in [0, 2**n_bits).

    Uses an 8-bit lookup table over successive byte slices rather than
    a bit-serial Python-level loop over ``n_bits`` - this constant-
    factor win is unconditional, independent of any sparsity
    assumption, and was confirmed via profiling to be one of the
    dense implementation's non-negligible costs at N=50/100.

    Args:
        values: Integer numpy array.
        n_bits: Number of bits to examine (values are assumed to fit).

    Returns:
        An integer array of the same shape as ``values``, containing
        the number of set bits in each element.
    """
    values = values.astype(np.uint32)
    count = np.zeros(values.shape, dtype=np.int64)
    for shift in range(0, n_bits, 8):
        count += _POPCOUNT_BYTE_LUT[(values >> shift) & 0xFF]
    return count

def _phase_from_popcount(
    values: NDArray[np.integer], n_bits: int, conjugate: bool = False
) -> NDArray[np.complexfloating]:
    """``1j ** popcount(values)``, without the complex exponentiation.

    Two bit-level shortcuts over the obvious
    ``1j ** _popcount_array(values, n_bits)``:

    1. **The accumulator is ``uint8``, not ``int64``.** Only
       ``popcount & 3`` affects the result, and ``uint8`` arithmetic
       wraps modulo 256. Since ``256 % 4 == 0``, wrapping cannot
       change the low two bits: ``(s % 256) & 3 == s & 3`` for every
       ``s``. So the narrow accumulator is exact here, not an
       approximation - it just moves 8x less data per byte-slice.
    2. **The power becomes a 4-entry table lookup.** ``i**k`` for
       ``k & 3`` is one of ``{1, i, -1, -i}``.

    Measured together on 4M terms at ``n_bits=14``: 133.8ms -> 50.6ms,
    a 2.65x improvement on this step, output bit-identical to the
    exponentiation form.

    Args:
        values: Integer array whose set bits are to be counted
            (typically ``x & z``).
        n_bits: Number of bits to examine.
        conjugate: If ``True``, return ``conj(1j ** popcount(...))``
            by gathering from the pre-conjugated table instead of
            building a ``np.conj`` temporary of the result - the form
            the coefficient formula needs. Free: conjugation of a
            fourth root of unity is just a different table.

    Returns:
        A complex array of the same shape, each entry in
        ``{1, i, -1, -i}``.
    """
    narrowed = values.astype(np.uint32)
    count = np.zeros(narrowed.shape, dtype=np.uint8)
    for shift in range(0, n_bits, 8):
        count += _POPCOUNT_BYTE_LUT[(narrowed >> shift) & 0xFF]
    table = _CONJ_PHASE_BY_POPCOUNT_MOD4 if conjugate else _PHASE_BY_POPCOUNT_MOD4
    return table[count & 3]


def _coefficients_from_transformed(
    transformed: NDArray[np.complexfloating],
    row_x: NDArray[np.integer],
    z_indices: NDArray[np.integer],
    n_qubits: int,
    inv_dim: float,
    atol: float,
) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.complexfloating]]:
    """Phase, scale, threshold and emit one transformed chunk.

    Turns a ``(rows, dim)`` transformed block into the thresholded
    ``(x, z, coefficient)`` triples the pipeline accumulates, applying
    ``conj(i**popcount(x & z)) / dim`` on the way.

    Uses the compiled ``coeffs_native`` kernel when available. That
    kernel fuses what NumPy must do as five separate passes - phase
    factor, scale, threshold, index gather, dtype narrowing - into one
    pass that keeps each value in registers, measured at 77.6us
    against 614.9us for the NumPy stages on a real N=150 chunk (7.9x).
    The NumPy path below is kept as an exact fallback.

    Both paths bound their working set by ``rows * dim`` - i.e. by
    ``chunk_size`` - never by ``dim**2`` or the total term count, and
    neither shares state between chunks.
    """
    if _coeffs_native is not None and transformed.dtype == np.complex128:
        block = np.ascontiguousarray(transformed)
        return _coeffs_native.coeffs_from_transformed(block, row_x, atol)

    conj_phase = _phase_from_popcount(
        row_x[:, np.newaxis] & z_indices, n_qubits, conjugate=True
    )
    coefficients = transformed * (conj_phase * inv_dim)
    row_idx, z_idx = np.nonzero(np.abs(coefficients) > atol)
    return row_x[row_idx], z_idx, coefficients[row_idx, z_idx]


def _walsh_hadamard_transform_rows(
    array: NDArray[np.complexfloating],
    overwrite_input: bool = False,
) -> NDArray[np.complexfloating]:
    """Apply the unnormalized Walsh-Hadamard Transform along axis 1.

    Uses the standard in-place butterfly algorithm: at each of
    log2(dim) stages, pairs of elements a distance ``h`` apart are
    replaced by their sum and difference. No normalization is applied
    here (that is folded into the phase-factor step); this matches
    the convention (H^{\\otimes n})^2 = 2^n * I^{\\otimes n}.

    Args:
        array: A 2D complex array of shape ``(rows, dim)`` where
            ``dim`` is a power of two. Each row is transformed
            independently.
        overwrite_input: If ``False`` (default), ``array`` is copied
            first so the caller's array is left untouched - safe for
            any caller. If ``True``, the transform is applied directly
            to ``array`` (still returned, for API symmetry with the
            ``False`` case) without an extra full-size copy; only pass
            ``True`` when the caller no longer needs ``array`` in its
            original form after this call (e.g. it was gathered solely
            to be transformed) - at N=150 this copy alone is ~2.73GiB,
            found to be the dominant memory bottleneck for large N,
            upstream of and separate from the dense-output cost
            ``fwht_pauli_coefficients(..., sparse=True)`` avoids.

    Returns:
        An array of the same shape with the transform applied to each
        row - a new array if ``overwrite_input=False``, otherwise
        ``array`` itself (mutated in place).
    """
    transformed = array if overwrite_input else array.copy()
    dim = array.shape[1]
    rows = array.shape[0]

    # Compiled path. The NumPy butterfly below
    # operates on stride-2 views, which cannot be vectorized; the C
    # kernel runs the same stages over contiguous memory and blocks
    # them so each tile is carried through log2(tile) stages while
    # cache-resident. Identical output - the tests assert it is
    # bit-identical - so this is purely a speed path, and its absence
    # changes results not at all.
    #
    # The tile size is derived from this machine's L1 data cache
    # (sysconf) once per distinct dim, not re-derived on every call -
    # see _wht_tile_for_dim's own comment for the process-lifetime
    # cache and why it matters more at HPC scale than a purely local
    # measurement suggests. Passing tile=0 (the kernel's own "derive
    # it yourself" sentinel) would give the same numeric result but
    # re-run the sysconf-based derivation inside the kernel on every
    # single chunk - this hoists that to once per dim per process.
    if (
        _wht_native is not None
        and transformed.dtype == np.complex128
        and transformed.flags["C_CONTIGUOUS"]
        and rows > 0
        and dim > 1
    ):
        _wht_native.wht_rows_inplace(transformed, _wht_tile_for_dim(dim))
        return transformed

    # ONE scratch buffer, allocated once and reused across all
    # log2(dim) stages, sized for the largest half-block (dim // 2
    # per row). The obvious spelling of the butterfly,
    #
    #     left, right = left + right, left - right
    #
    # allocates TWO full-size temporaries per stage and reads each
    # operand twice - roughly four passes over the data and two heap
    # allocations, every stage. Profiling attributed 53% of the
    # chunked path's runtime to this function, and replacing that
    # idiom with the in-place form below measured 1.29-1.32x faster
    # on the transform in isolation, output verified identical.
    #
    # The ordering matters and is not interchangeable: the difference
    # must be computed into scratch BEFORE ``left`` is overwritten,
    # because ``np.add(left, right, out=left)`` destroys the operand
    # the subtraction needs.
    scratch = np.empty((rows, dim // 2), dtype=transformed.dtype)

    span = 1
    while span < dim:
        blocks = dim // (2 * span)
        view = transformed.reshape(rows, blocks, 2, span)
        left = view[:, :, 0, :]
        right = view[:, :, 1, :]
        half = scratch[:, : blocks * span].reshape(rows, blocks, span)
        np.subtract(left, right, out=half)   # difference -> scratch
        np.add(left, right, out=left)        # sum in place
        right[...] = half                    # difference back
        transformed = view.reshape(rows, dim)
        span *= 2
    return transformed


class _GrowableArray:
    """Amortized-doubling growable 1-D array.

    Appending ``n`` elements at a time and doubling capacity on
    overflow costs O(total elements) amortized, rather than O(chunks)
    reallocations of ever-larger arrays (the naive ``np.concatenate``
    per chunk) or O(total elements) Python-object overhead (a plain
    list of scalars) - keeps the chunked path's per-chunk append cheap
    relative to that chunk's O(chunk_size * dim * log dim) transform
    cost, rather than trading space complexity for a new
    time-complexity regression.
    """

    def __init__(self, dtype, initial_capacity: int = 1024):
        self._data = np.empty(initial_capacity, dtype=dtype)
        self._size = 0

    def extend(self, values: NDArray) -> None:
        n = len(values)
        if n == 0:
            return
        needed = self._size + n
        if needed > len(self._data):
            new_capacity = max(needed, len(self._data) * 2)
            grown = np.empty(new_capacity, dtype=self._data.dtype)
            grown[: self._size] = self._data[: self._size]
            self._data = grown
        self._data[self._size : self._size + n] = values
        self._size += n

    def finalize(self) -> NDArray:
        return self._data[: self._size]


def _checkpoint_progress_path(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + ".progress.json")


def _parallel_checkpoint_progress_path(checkpoint_path: str | Path) -> Path:
    """Progress-file path for the *parallel* checkpoint format -
    deliberately a different filename from
    ``_checkpoint_progress_path``'s sequential format, so the two never
    collide or silently misinterpret each other's progress file if a
    caller reuses the same ``checkpoint_path`` between
    ``fwht_pauli_terms_iter`` and ``parallel_decompose``.
    """
    return Path(str(checkpoint_path) + ".parallel_progress.json")


# Persisted on disk: one record per completed chunk. u64 matches the
# width the frame header already uses for chunk_index, so the two
# cannot disagree about range. Never change the width or endianness.
_PROGRESS_RECORD = struct.Struct("<Q")


def _append_progress_record(
    progress_path: str | Path, chunk_index: int
) -> None:
    """Record one completed chunk by appending a fixed-width record.

    This replaced a ``json.dump`` of the whole completed set on every
    chunk, which was O(n) per chunk and so O(n^2) across a run. At
    N=150's 5,595 chunks that marker cost 15.13x the payload frame
    write by the final chunk and ~91% of the per-chunk checkpoint
    total. An 8-byte append is O(1) - measured flat at ~13-21us
    across a 20,000x range of completed counts.
    """
    with open(progress_path, "ab") as f:
        f.write(_PROGRESS_RECORD.pack(chunk_index))


def _read_completed_indices(progress_path: str | Path) -> set[int]:
    """Recover the set of completed chunk indices.

    Reads whole records only: a trailing partial record is DISCARDED
    rather than treated as corruption. Appends are 8 bytes written in
    one call and never rewritten, so only the final record can ever be
    torn, and a torn tail means exactly "that chunk was not recorded" -
    the chunk is resubmitted and recomputed, which is the pre-existing
    contract.

    Returns a set, not a sequence: records are not sorted (parallel
    workers complete out of order) and may be duplicated (a
    rollback-resume re-records chunks it recomputes).
    """
    progress_path = Path(progress_path)
    if not progress_path.exists():
        return set()
    data = progress_path.read_bytes()
    size = _PROGRESS_RECORD.size
    n_whole = len(data) // size
    return {
        _PROGRESS_RECORD.unpack_from(data, i * size)[0]
        for i in range(n_whole)
    }


_CHECKPOINT_MAGIC = b"PKCP"
_CHECKPOINT_VERSION = 1

# magic, version, index-dtype code, reserved, chunk index, term count.
# chunk_index and n_terms are u64 deliberately: N=150 alone has
# 91,652,096 terms across thousands of chunks, and a u32 term count
# would cap a single frame at 4.29e9 - close enough to real workloads
# to be a latent bug rather than a safe assumption.
_CHECKPOINT_HEADER_STRUCT = struct.Struct("<4sHBBQQ")

# Persisted on disk: these codes are part of the format and must never
# be renumbered. _index_dtype_for_dim picks the width from `dim`, so a
# file can legitimately contain any of the three.
_INDEX_DTYPE_CODES: dict[int, np.dtype] = {
    0: np.dtype(np.uint16),
    1: np.dtype(np.uint32),
    2: np.dtype(np.intp),
}
_INDEX_DTYPE_TO_CODE = {v: k for k, v in _INDEX_DTYPE_CODES.items()}


def _checkpoint_frame_header(
    chunk_index: int, n_terms: int, idx_dtype: np.dtype
) -> bytes:
    """Pack one frame's fixed-width header.

    The index dtype is recorded per frame rather than assumed, which is
    a correctness requirement and not a size optimization:
    ``_index_dtype_for_dim`` returns ``uint16`` only while
    ``dim <= 65536``, so a reader that assumed a width would silently
    WRAP a 17-qubit-or-larger operator's indices and produce wrong
    Pauli labels instead of an error.
    """
    code = _INDEX_DTYPE_TO_CODE.get(np.dtype(idx_dtype))
    if code is None:
        raise ValueError(
            f"unsupported checkpoint index dtype {idx_dtype!r} - "
            f"expected one of {sorted(str(d) for d in _INDEX_DTYPE_CODES.values())}"
        )
    return _CHECKPOINT_HEADER_STRUCT.pack(
        _CHECKPOINT_MAGIC, _CHECKPOINT_VERSION, code, 0, chunk_index, n_terms
    )


def _parse_checkpoint_frame_header(raw: bytes) -> tuple[int, int, np.dtype]:
    """Unpack a frame header, validating every field.

    Returns ``(chunk_index, n_terms, idx_dtype)``. Raises ``ValueError``
    on a short buffer, wrong magic, unknown version, or unknown index
    dtype - all of which mean the file is not a frame boundary this
    reader understands, and guessing would corrupt the replay.
    """
    if len(raw) < _CHECKPOINT_HEADER_STRUCT.size:
        raise ValueError(
            f"truncated checkpoint frame header: got {len(raw)} bytes, "
            f"need {_CHECKPOINT_HEADER_STRUCT.size}"
        )
    magic, version, code, _reserved, chunk_index, n_terms = (
        _CHECKPOINT_HEADER_STRUCT.unpack(raw[: _CHECKPOINT_HEADER_STRUCT.size])
    )
    if magic != _CHECKPOINT_MAGIC:
        raise ValueError(
            f"bad checkpoint frame magic {magic!r} - expected "
            f"{_CHECKPOINT_MAGIC!r}"
        )
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version {version} - this "
            f"build writes and reads version {_CHECKPOINT_VERSION}"
        )
    idx_dtype = _INDEX_DTYPE_CODES.get(code)
    if idx_dtype is None:
        raise ValueError(
            f"unknown checkpoint index dtype code {code}"
        )
    return chunk_index, n_terms, idx_dtype


def _append_checkpoint_frame(
    checkpoint_path: str | Path,
    chunk_index: int,
    x_out: NDArray[np.integer],
    z_out: NDArray[np.integer],
    coeff_out: NDArray[np.complexfloating],
    idx_dtype: np.dtype,
) -> None:
    """Append one chunk as a single self-describing frame.

    Deliberately free of any per-term Python work. The JSONL format
    this replaces cost ~4.2 us per term in ``.tolist()`` + a dict
    literal + ``json.dumps``, GIL-held in the parent's drain loop -
    measured at 94.8% of the writer's total cost, versus 1.1% for the
    disk I/O itself. Three ``tobytes()`` calls move the same
    information with no object churn, which is what keeps the drain
    loop empty enough for the parallel path to scale.
    """
    x_arr = np.ascontiguousarray(x_out, dtype=idx_dtype)
    z_arr = np.ascontiguousarray(z_out, dtype=idx_dtype)
    coeff_arr = np.ascontiguousarray(coeff_out, dtype=complex)
    if not (len(x_arr) == len(z_arr) == len(coeff_arr)):
        raise ValueError(
            f"checkpoint frame arrays must be equal length, got "
            f"{len(x_arr)}, {len(z_arr)}, {len(coeff_arr)}"
        )
    header = _checkpoint_frame_header(chunk_index, len(x_arr), idx_dtype)
    with open(checkpoint_path, "ab") as f:
        f.write(header)
        f.write(x_arr.tobytes())
        f.write(z_arr.tobytes())
        f.write(coeff_arr.tobytes())


def _iter_checkpoint_frames(
    checkpoint_path: str | Path,
    valid_indices: set[int] | None = None,
) -> Iterator[tuple[int, NDArray, NDArray, NDArray]]:
    """Yield ``(chunk_index, x, z, coeff)`` per complete frame.

    One frame is live at a time, so replay memory is bounded by the
    largest chunk rather than by the file - the property the JSONL
    reader lacked. That reader called ``f.readlines()`` and built three
    full Python lists plus a dedup dict, which at N=150 is >20 GB and
    simply could not run.

    Stops cleanly at the first frame that is incomplete (a crash
    mid-write) or absent from ``valid_indices`` (recorded but never
    marked complete). Because a reader stops at the first such frame,
    a duplicate can never survive to be read, which is what removes
    the need for the old ``last_by_key`` deduplication entirely.
    Corruption that is not a clean truncation - a bad magic in an
    earlier frame - still raises, since that is real damage rather
    than an interrupted append.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return

    header_size = _CHECKPOINT_HEADER_STRUCT.size
    with open(checkpoint_path, "rb") as f:
        while True:
            raw_header = f.read(header_size)
            if not raw_header:
                return  # clean end of file
            if len(raw_header) < header_size:
                return  # torn header from an interrupted append
            chunk_index, n_terms, idx_dtype = _parse_checkpoint_frame_header(
                raw_header
            )
            idx_nbytes = n_terms * idx_dtype.itemsize
            coeff_nbytes = n_terms * np.dtype(complex).itemsize
            payload = f.read(2 * idx_nbytes + coeff_nbytes)
            if len(payload) < 2 * idx_nbytes + coeff_nbytes:
                return  # torn payload from an interrupted append
            if valid_indices is not None and chunk_index not in valid_indices:
                return
            x = np.frombuffer(payload, dtype=idx_dtype, count=n_terms)
            z = np.frombuffer(
                payload, dtype=idx_dtype, count=n_terms, offset=idx_nbytes
            )
            coeff = np.frombuffer(
                payload, dtype=complex, count=n_terms, offset=2 * idx_nbytes
            )
            yield chunk_index, x, z, coeff


def _load_parallel_checkpoint(
    checkpoint_path: str | Path | None,
) -> tuple[set[int], Iterator[tuple[NDArray, NDArray, NDArray]] | None]:
    """Read an existing *parallel* checkpoint, if any.

    The progress file is append-only: one 8-byte little-endian
    ``u64`` record per completed chunk, in whatever order workers
    finished it (see ``_read_completed_indices``, which reads whole
    records into a set and discards a torn trailing record). Unlike
    the sequential path, which derives a single monotonic
    ``next_chunk`` from the same kind of file (correct only when
    chunks complete strictly in order), parallel workers complete
    chunks in whatever order the pool schedules them - so resume uses
    the recovered set directly, skipping exactly those indices and
    re-submitting every other chunk regardless of completion order.

    Returns ``(completed_chunk_indices, frames | None)``, where
    ``frames`` is a lazy iterator of ``(x, z, coeff)`` triples, one per
    recorded frame. It is an iterator rather than one combined tuple so
    that replay holds one chunk at a time; materializing the whole
    checkpoint is what made resume unusable at N=150.
    """
    if checkpoint_path is None:
        return set(), None
    checkpoint_path = Path(checkpoint_path)
    progress_path = _parallel_checkpoint_progress_path(checkpoint_path)
    if not checkpoint_path.exists() or not progress_path.exists():
        return set(), None

    completed = _read_completed_indices(progress_path)
    if not completed:
        return completed, None

    def _frames() -> Iterator[tuple[NDArray, NDArray, NDArray]]:
        for _index, x, z, coeff in _iter_checkpoint_frames(
            checkpoint_path, valid_indices=completed
        ):
            yield x, z, coeff

    return completed, _frames()


def _append_parallel_checkpoint_chunk(
    checkpoint_path: str | Path,
    completed_chunk_indices: set[int],
    chunk_index: int,
    x_out: NDArray[np.integer],
    z_out: NDArray[np.integer],
    coeff_out: NDArray[np.complexfloating],
    idx_dtype: np.dtype,
) -> None:
    """Append one completed chunk's frame, then record its index.

    Called from the main process only (after collecting a worker's
    result), so this file/set update is never concurrently written by
    multiple processes - workers return their chunk's triples to the
    main process; they do not write the checkpoint file directly.

    The progress record is a single 8-byte little-endian ``u64``
    holding ``chunk_index``, appended to the progress file
    (``_append_progress_record``) - an O(1) write, unlike the earlier
    rewrite-the-whole-set-as-JSON marker it replaced.

    The frame is written before the progress marker is appended, so a
    crash mid-write leaves the progress file not yet listing this
    chunk: the frame is then either torn (and stops the reader) or
    complete-but-unmarked (and is dropped by ``valid_indices``). Either
    way the chunk is simply resubmitted on resume, and no duplicate can
    reach a reader - which is why this format needs no deduplication.
    """
    _append_checkpoint_frame(
        checkpoint_path, chunk_index, x_out, z_out, coeff_out, idx_dtype
    )
    completed_chunk_indices.add(chunk_index)
    progress_path = _parallel_checkpoint_progress_path(checkpoint_path)
    _append_progress_record(progress_path, chunk_index)


def _load_checkpoint(
    checkpoint_path: str | Path | None,
) -> tuple[int, Iterator[tuple[NDArray, NDArray, NDArray]] | None]:
    """Read an existing *sequential* checkpoint, if any.

    Returns ``(resume_from_chunk_index, frames | None)``: the chunk
    index to resume from (0 if absent or incomplete) and a lazy
    iterator of previously-recorded ``(x, z, coeff)`` triples, one per
    frame, or ``None`` if there is nothing to replay.

    The progress file uses the same append-only record format as the
    parallel path - one 8-byte little-endian ``u64`` per completed
    chunk (see ``_read_completed_indices``, which reads whole records
    into a set and discards a torn trailing record). Sequential chunks
    complete strictly in order, so the recovered set is contiguous and
    ``next_chunk`` is derived as ``max(recovered) + 1`` rather than
    stored directly - every frame before ``next_chunk`` is valid, and
    any frame at or beyond it was written but never marked.
    """
    if checkpoint_path is None:
        return 0, None
    checkpoint_path = Path(checkpoint_path)
    progress_path = _checkpoint_progress_path(checkpoint_path)
    if not checkpoint_path.exists() or not progress_path.exists():
        return 0, None

    completed = _read_completed_indices(progress_path)
    if not completed:
        return 0, None
    # Sequential chunks complete strictly in order, so the recovered
    # set is contiguous and the resume point is one past its maximum.
    next_chunk = max(completed) + 1

    def _frames() -> Iterator[tuple[NDArray, NDArray, NDArray]]:
        for _index, x, z, coeff in _iter_checkpoint_frames(
            checkpoint_path, valid_indices=set(range(next_chunk))
        ):
            yield x, z, coeff

    return next_chunk, _frames()


def _append_checkpoint_chunk(
    checkpoint_path: str | Path,
    next_chunk: int,
    x_out: NDArray[np.integer],
    z_out: NDArray[np.integer],
    coeff_out: NDArray[np.complexfloating],
    idx_dtype: np.dtype,
) -> None:
    """Append one completed chunk's frame, then advance the marker.

    ``next_chunk`` is the count of completed chunks, so the frame being
    written carries index ``next_chunk - 1``. The marker advance is a
    single 8-byte little-endian ``u64`` record holding that index,
    appended to the progress file (``_append_progress_record``) - an
    O(1) write, unlike the earlier rewrite-the-whole-marker-as-JSON
    approach it replaced. The frame is written before the marker
    advances, so a crash mid-write leaves a torn or unmarked frame
    that the reader drops, and that chunk is recomputed on resume
    rather than silently corrupted.
    """
    _append_checkpoint_frame(
        checkpoint_path, next_chunk - 1, x_out, z_out, coeff_out, idx_dtype
    )
    progress_path = _checkpoint_progress_path(checkpoint_path)
    # next_chunk is the COUNT of completed chunks, so the chunk just
    # written carries index next_chunk - 1 - matching the index passed
    # to _append_checkpoint_frame immediately above.
    _append_progress_record(progress_path, next_chunk - 1)


class NullChunkSink:
    """No-op chunk sink used when the caller does not want on-disk output.

    Drain loops always call ``write_*`` on a sink so the hot path stays
    branchless with respect to ``checkpoint_path is not None``. Workers
    never see this object: only the main-thread drain writes.
    """

    __slots__ = ()

    def write_sequential_chunk(
        self,
        next_chunk: int,
        x_out: NDArray[np.integer],
        z_out: NDArray[np.integer],
        coeff_out: NDArray[np.complexfloating],
        idx_dtype: np.dtype,
    ) -> None:
        return None

    def write_parallel_chunk(
        self,
        completed_chunk_indices: set[int],
        chunk_index: int,
        x_out: NDArray[np.integer],
        z_out: NDArray[np.integer],
        coeff_out: NDArray[np.complexfloating],
        idx_dtype: np.dtype,
    ) -> None:
        return None


class CheckpointChunkSink:
    """Main-thread PKCP writer for streamed ``(x, z, coeff)`` chunks.

    Same binary format as ``checkpoint_path=`` / CLI ``--write-chunks``:
    framed symplectic indices plus complex coefficients. Resume uses the
    sibling progress file. Never called from worker threads/processes.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = path

    def write_sequential_chunk(
        self,
        next_chunk: int,
        x_out: NDArray[np.integer],
        z_out: NDArray[np.integer],
        coeff_out: NDArray[np.complexfloating],
        idx_dtype: np.dtype,
    ) -> None:
        _append_checkpoint_chunk(
            self._path, next_chunk, x_out, z_out, coeff_out, idx_dtype
        )

    def write_parallel_chunk(
        self,
        completed_chunk_indices: set[int],
        chunk_index: int,
        x_out: NDArray[np.integer],
        z_out: NDArray[np.integer],
        coeff_out: NDArray[np.complexfloating],
        idx_dtype: np.dtype,
    ) -> None:
        _append_parallel_checkpoint_chunk(
            self._path,
            completed_chunk_indices,
            chunk_index,
            x_out,
            z_out,
            coeff_out,
            idx_dtype,
        )


def _make_chunk_sink(checkpoint_path: str | Path | None) -> NullChunkSink | CheckpointChunkSink:
    """Build the drain-loop sink once (branch only at setup)."""
    if checkpoint_path is None:
        return NullChunkSink()
    return CheckpointChunkSink(checkpoint_path)


def iter_checkpoint_chunks(
    path: str | Path,
) -> Iterator[tuple[int, NDArray, NDArray, NDArray]]:
    """Yield ``(chunk_index, x, z, coeff)`` from a written chunk file.

    Public reader for files produced by ``checkpoint_path=`` on the
    library APIs or by ``paulikit decompose --write-chunks PATH`` /
    ``--checkpoint-path PATH``. One frame is live at a time. Arrays are
    symplectic ``x``/``z`` indices and ``complex`` coefficients - use
    ``terms_from_arrays`` only for subsets that need Pauli-string labels.
    """
    yield from _iter_checkpoint_frames(path)


def count_stream_chunks(
    operator,
    chunk_size: int,
    *,
    assume_dense: bool = False,
) -> int:
    """Return how many streamed chunks ``chunk_size`` will produce.

    Same prep as the streamed/parallel drains. The CLI always calls
    this once before the timed ``--parallel`` / ``--stream`` drain so
    quiet and ``--progress`` paths share a CPU warm-up; ``--progress``
    also uses the count for ``k/N``. Library callers that only need
    the count without draining can call this directly; drains that
    already expose ``chunk_plan=`` can fill ``n_chunks`` during their
    own prep instead.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    (
        _operator,
        is_sparse_input,
        dim,
        _n_qubits,
        _p_nz,
        _q_nz,
        x_nz,
        _values_nz,
    ) = _prepare_operator_for_fwht(operator, assume_dense=assume_dense)
    active_x, _inverse = _active_x_and_inverse(
        x_nz, dim, is_fully_dense=assume_dense and not is_sparse_input
    )
    n_active = len(active_x)
    if n_active == 0:
        return 0
    return (n_active + chunk_size - 1) // chunk_size


def _iter_chunked_coefficients(
    operator,
    is_sparse_input: bool,
    active_x: NDArray[np.intp],
    inverse: NDArray[np.intp],
    p_nz: NDArray[np.intp],
    q_nz: NDArray[np.intp],
    values_nz: NDArray[np.complexfloating],
    dim: int,
    n_qubits: int,
    n_active: int,
    z_indices: NDArray[np.intp],
    chunk_size: int,
    atol: float,
    checkpoint_path: str | Path | None,
    chunk_plan: dict | None = None,
):
    """Generator over chunks of already-thresholded ``(x, z,
    coefficient)`` triples - the shared tile-producing core of the
    chunked path, used by both ``fwht_pauli_coefficients`` (which
    accumulates every tile into one COO triple) and
    ``fwht_pauli_terms_iter`` (which converts each tile to labels and
    yields it immediately). Each chunk is a fully independent
    sub-problem (divide-and-conquer: no cross-chunk combination step
    exists in the underlying math, unlike e.g. tiled matrix
    multiply's block-sum reduction) - this generator is what keeps
    that independence visible to callers instead of re-fusing every
    tile before returning, which is the actual fix streaming needed.

    Yields ``(chunk_x, chunk_z, chunk_coeff)`` - three 1-D arrays of
    equal length, one triple per chunk, in chunk order.

    If ``chunk_plan`` is a dict, it receives ``n_chunks`` (planned
    tile count) once the chunk grid is known, before any yield - so a
    CLI progress helper can show ``k/N`` without a second prep pass.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    # dim is a power of two, so 1/dim is exact in binary floating
    # point and multiplying by it is bit-identical to dividing by dim
    # - but it hoists the reciprocal out of the per-chunk loop and
    # turns a per-term division into a per-term multiply.
    inv_dim = 1.0 / dim

    # FULLY DENSE fast path (same as _parallel_worker_chunk /
    # parallel_decompose_arrays - see their own comments for the full
    # rationale, the structural proof of why this is always correct,
    # and the measurement behind it). When every cell of `operator` is
    # nonzero, active_x is provably arange(dim) and each chunk can be
    # gathered directly as operator[x^q, q] with no dependency on any
    # global sort at all - skips the O(dim**2 log dim) argsort AND the
    # per-chunk np.zeros-and-scatter entirely. This generator is the
    # shared body `fwht_pauli_coefficients`/`fwht_pauli_terms_iter`
    # both call - before this fix, NEITHER of those public entry
    # points had received any of this optimization, unlike
    # parallel_decompose_arrays and parallel_decompose, which did
    # earlier in the same investigation - a real, measured gap (perf
    # stat instructions:u): the unfixed sequential path cost ~6.48x
    # pauli_lcu's own decomposition, vs ~1.95x for the already-fixed
    # parallel path, on identical dense random Hermitian input.
    is_fully_dense = (not is_sparse_input) and n_active == dim
    if is_fully_dense:
        sorted_inverse = sorted_p_nz = sorted_q_nz = sorted_values = None
    else:
        # inverse is sorted first so each chunk's nonzero entries
        # (p_nz[lo:hi], q_nz[lo:hi]) are a contiguous slice, found via
        # searchsorted on chunk boundaries - avoiding an O(nnz) boolean
        # mask per chunk. kind="quicksort", not "stable" - see
        # parallel_decompose_arrays's own comment: no downstream use
        # needs original-order preservation within a bucket of equal
        # x-values.
        order = np.argsort(inverse, kind="quicksort")
        sorted_inverse = inverse[order]
        sorted_p_nz = p_nz[order]
        sorted_q_nz = q_nz[order]
        # The values follow the same permutation, so each chunk's slice
        # is contiguous and no per-chunk operator lookup is needed.
        sorted_values = values_nz[order]

    chunk_starts = list(range(0, n_active, chunk_size))
    if chunk_plan is not None:
        chunk_plan["n_chunks"] = len(chunk_starts)
    resume_from, checkpoint_frames = _load_checkpoint(checkpoint_path)
    idx_dtype = _index_dtype_for_dim(dim)
    # Construct once: drain always calls sink.write_* (NullChunkSink when
    # checkpoint_path is None) so the per-chunk path stays branchless.
    chunk_sink = _make_chunk_sink(checkpoint_path)
    if checkpoint_frames is not None and resume_from > 0:
        # Replay already-completed chunks from the checkpoint rather
        # than recomputing them. Frames record their own chunk index,
        # so this replays per original chunk - the JSONL format could
        # not preserve boundaries and had to yield one combined tile.
        for frame in checkpoint_frames:
            yield frame

    for chunk_index in range(resume_from, len(chunk_starts)):
        chunk_start = chunk_starts[chunk_index]
        chunk_end = min(chunk_start + chunk_size, n_active)

        if is_fully_dense:
            if _gather_native is not None:
                gathered_chunk = _gather_native.gather_dense_chunk(
                    operator, chunk_start, chunk_end - chunk_start
                )
            else:
                x_values = np.arange(chunk_start, chunk_end)
                q_range = z_indices[0]
                p_indices = x_values[:, np.newaxis] ^ q_range[np.newaxis, :]
                gathered_chunk = np.ascontiguousarray(
                    operator[p_indices, q_range[np.newaxis, :]]
                )
        else:
            lo = int(np.searchsorted(sorted_inverse, chunk_start))
            hi = int(np.searchsorted(sorted_inverse, chunk_end))

            gathered_chunk = np.zeros((chunk_end - chunk_start, dim), dtype=complex)
            # A slice of the pre-extracted values, not a fresh operator
            # lookup: the scipy fancy-index call this replaces cost 45.4us
            # per chunk to fetch ~64 values at the N=150 shape - CSR index
            # validation overhead, paid thousands of times.
            gathered_values = sorted_values[lo:hi]
            gathered_chunk[
                sorted_inverse[lo:hi] - chunk_start, sorted_q_nz[lo:hi]
            ] = gathered_values

        transformed_chunk = _walsh_hadamard_transform_rows(
            gathered_chunk, overwrite_input=True
        )

        chunk_x_out, z_idx, chunk_coeff_out = _coefficients_from_transformed(
            transformed_chunk, active_x[chunk_start:chunk_end],
            z_indices, n_qubits, inv_dim, atol,
        )

        chunk_sink.write_sequential_chunk(
            chunk_index + 1,
            chunk_x_out, z_idx, chunk_coeff_out, idx_dtype,
        )

        yield chunk_x_out, z_idx, chunk_coeff_out


def _prepare_operator_for_fwht(operator, assume_dense=False):
    """Shared validation/setup for ``fwht_pauli_coefficients`` and
    ``fwht_pauli_terms_iter``: shape/power-of-two checks, sparse-input
    detection and CSR conversion, and the XOR-index gather's raw
    nonzero-entry arrays (before deduplicating into ``active_x``,
    which each caller does slightly differently downstream).

    ``assume_dense`` skips the ``np.nonzero`` sparsity-DETECTION scan
    entirely when ``True`` and ``operator`` is a plain ndarray (never
    for scipy.sparse input) - not a correctness-risking promise, a
    pure performance choice. ``p_nz``/``q_nz``/``x_nz``/``values_nz``
    are returned as empty placeholders in that case, because
    ``parallel_decompose_arrays``'s fully-dense fast path (see
    ``_parallel_worker_chunk``) never uses them: it reads
    ``operator[p, q]`` directly per chunk instead of scattering from a
    precomputed nonzero list, so it is correct for ANY actual sparsity
    pattern, including one with exact-zero entries or even a mostly-
    zero operator - verified directly (an operator with a zeroed
    single cell, a zeroed row/column, and a 90%-zero random pattern
    all produced bit-identical output to auto-detection). Setting this
    when you already know your ndarray is worth reading in full (e.g.
    you just built it as ``rng.standard_normal(...)`` yourself, or any
    other case where its own sparsity is not worth discovering) skips
    only the DETECTION cost, which is real and measured: `perf stat`
    `instructions:u` showed `np.nonzero` alone costs ~594-688M
    instructions at dim=1024 - comparable to `pauli_lcu`'s entire
    decomposition (689M instructions) at the same dim, a structural
    cost paulikit pays to discover sparsity that a dense-only tool
    like `pauli_lcu` never pays. The only real downside of setting
    this on a GENUINELY sparse operator is performance, not
    correctness: the fast path reads and transforms every cell
    (dim*dim work) rather than skipping structurally-zero rows, so it
    is worthwhile only when the operator is dense enough that
    ``np.nonzero``'s scan cost is not worth paying to find that out;
    auto-detection (the default) remains the right choice whenever
    sparsity is unknown or expected.

    Returns ``(operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz)``.
    """
    dim = operator.shape[0]
    if operator.shape != (dim, dim):
        raise ValueError(f"operator must be square, got shape {operator.shape}")
    n_qubits = int(round(np.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError(
            f"operator dimension {dim} is not a power of two; "
            "pad it first with paulikit.hamiltonian.pad_to_power_of_two"
        )

    # operator may itself be a scipy.sparse matrix (e.g. from
    # paulikit.hamiltonian.build_hamiltonian(..., sparse=True)) - kept
    # sparse rather than densified here, since densifying+upcasting to
    # complex is exactly the ~4GiB N=150 cost this input path exists
    # to avoid. np.nonzero() and fancy indexing both dispatch
    # correctly to scipy.sparse's own implementations without
    # densifying (verified directly), except that sparse fancy
    # indexing returns a
    # numpy.matrix of shape (1, nnz) rather than a flat (nnz,) array,
    # which is why callers' gathers go through np.asarray(...).ravel().
    # A plain ndarray is never a scipy.sparse matrix - checking that
    # first, with no import at all, is what makes the scipy.sparse
    # import genuinely lazy for the (overwhelming majority) common
    # case rather than merely moved from module-import time to
    # first-call time. Only an operator that is NOT a plain ndarray
    # (e.g. actually IS scipy.sparse, or some other array-like) pays
    # the import, and even then only once per process
    # (_scipy_sparse_module caches it, including the "not installed"
    # case).
    if isinstance(operator, np.ndarray):
        is_sparse_input = False
    else:
        sp_module = _scipy_sparse_module()
        is_sparse_input = sp_module is not None and sp_module.issparse(operator)
    if is_sparse_input:
        # COO (e.g. from pad_to_power_of_two(..., sparse=True)) does
        # not support fancy indexing (operator[p_nz, q_nz]) - CSR
        # does, and np.nonzero() dispatches efficiently on CSR too.
        operator = operator.tocsr().astype(complex)
    else:
        # ascontiguousarray, not asarray: gather_native's compiled
        # fast path (see _parallel_worker_chunk) requires a
        # C-contiguous buffer - it reads the underlying doubles
        # directly with no NumPy-level stride handling, unlike the
        # general fancy-indexing fallback it replaces, which tolerates
        # any layout. A caller passing e.g. a Fortran-ordered dense
        # array (np.asfortranarray, or a transpose view) would
        # otherwise hit that requirement only when the kernel happens
        # to be built, a real gap found by direct testing - normalize
        # here, once per call, rather than per chunk or not at all.
        # A no-op copy (same memory layout already) whenever the
        # caller's array is already C-contiguous, which is NumPy's own
        # default and covers the overwhelming majority of real calls.
        operator = np.ascontiguousarray(operator, dtype=complex)

    if assume_dense and not is_sparse_input:
        # Caller-verified promise, not auto-detection: skip the
        # O(dim**2) np.nonzero scan (and the values_nz gather that
        # depends on it) entirely. p_nz/q_nz/x_nz/values_nz are all
        # returned as genuinely EMPTY placeholders -
        # parallel_decompose_arrays's fully-dense fast path reads
        # `operator` directly per chunk without ever touching these,
        # and _active_x_and_inverse's own dense fast path is now
        # selected by an explicit `is_fully_dense` flag passed by every
        # caller (== `assume_dense and not is_sparse_input`, computed
        # here and re-derived identically at each call site) rather
        # than inferred from `len(x_nz) == dim*dim` - so x_nz needs no
        # particular length at all here, only its dtype. Building
        # real dim*dim-length arrays here would itself cost real,
        # avoidable work (measured: even a scan-free
        # np.repeat/np.tile/XOR construction of x_nz alone still costs
        # ~583M instructions at dim=1024, close to np.nonzero's own
        # 688M - the O(dim**2) MATERIALIZATION is the real cost, not
        # np.nonzero's specific scan mechanics) AND, found later
        # (2026-09-20, while checking that this path never keeps a
        # second copy of the operator in memory): a real, measured
        # memory cost - the OLD `np.empty(dim*dim, dtype=np.intp)`
        # placeholder alone was 512 MiB at qubits=13, allocated and
        # counted in peak RSS despite never being read. Making
        # `_active_x_and_inverse`'s fast path explicit rather than
        # length-inferred is what makes a genuinely empty placeholder
        # safe here.
        p_nz = np.empty(0, dtype=np.intp)
        q_nz = np.empty(0, dtype=np.intp)
        x_nz = np.empty(0, dtype=np.intp)
        values_nz = np.empty(0, dtype=complex)
        return (
            operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
        )

    # Step 1: XOR-index gather, restricted to the operator's nonzero
    # entries. gathered[x, q] = operator[q ^ x, q] is nonzero only when
    # p = q ^ x is a nonzero entry of operator, i.e. x = p ^ q. Only
    # scattering those (x, q) cells - rather than gathering the full
    # dense (dim, dim) array via fancy indexing - avoids O(dim**2) work
    # for the O(N)-nonzero Hamiltonians this package targets (an
    # operator-sparsity-independent all-dense fallback would still be
    # correct here, but measurably slower on sparse input and no
    # faster on dense input).
    p_nz, q_nz = np.nonzero(operator)
    x_nz = p_nz ^ q_nz

    # Extract the nonzero VALUES once, here, rather than re-querying
    # the operator per chunk. Measured at the N=150 shape: one scipy
    # fancy-index call fetching a chunk's 64 values cost 45.4us -
    # about 700ns per value, and more than it costs to zero the whole
    # 512 KiB destination block. That is CSR index validation and
    # broadcasting overhead, paid once per chunk over thousands of
    # chunks, not data movement. Fetching all of them in one call and
    # slicing per chunk replaces every one of those calls with an
    # array slice.
    #
    # This costs O(nnz) memory - 45,000 complex values (~700 KiB) at
    # N=150, against the operator's own storage which the caller
    # already holds - so it does not change the pipeline's memory
    # profile, which is bounded by chunk_size * dim.
    values_nz = operator[p_nz, q_nz]
    if is_sparse_input:
        # scipy.sparse fancy indexing returns a numpy.matrix of shape
        # (1, nnz), not a flat (nnz,) ndarray - verified directly.
        values_nz = np.asarray(values_nz).ravel()

    return (
        operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
    )


def _active_x_and_inverse(x_nz, dim, is_fully_dense=False):
    """Shared by every caller downstream of ``_prepare_operator_for_fwht``:
    deduplicates ``x_nz`` into ``active_x`` (the distinct XOR masks) and
    ``inverse`` (per-nonzero-entry index into ``active_x``), the same
    pair ``np.unique(x_nz, return_inverse=True)`` returns.

    Takes a fast path for fully DENSE input, signalled explicitly by
    ``is_fully_dense`` (the caller already knows this - either because
    every cell of the (dim, dim) operator was verified nonzero via
    ``np.nonzero``, or because the caller passed ``assume_dense=True``
    as its own promise) rather than being inferred from
    ``len(x_nz) == dim * dim``. When true, ``active_x`` is provably
    ``arange(dim)`` and ``inverse`` is provably identical to the REAL
    XOR-gathered ``x_nz`` (when one was actually built) with no need
    to call ``np.unique`` (an internal sort over ``dim*dim`` elements)
    to discover either fact:

    For any fixed p, ``x = p ^ q`` is a bijection of q over ``[0, dim)``
    (XOR by a constant is its own inverse), so a fully-populated row p
    contributes every value in ``[0, dim)`` to ``x_nz`` exactly once.
    With all `dim` rows present, every value in ``[0, dim)`` therefore
    appears in ``x_nz`` exactly ``dim`` times - i.e. ``x_nz``'s value
    set is already ``{0, ..., dim-1}`` and each raw entry already IS
    its own index into that set. This is a structural guarantee of the
    XOR-gather construction, not a property of any particular matrix's
    values - true for any fully dense operator regardless of content.

    ``assume_dense=True`` callers never build a real ``x_nz`` at all
    (see ``_prepare_operator_for_fwht``'s own dense-fast-path branch) -
    they pass an empty placeholder and ``is_fully_dense=True`` directly,
    since ``inverse``'s CONTENTS are never read on the fully-dense path
    downstream (``parallel_decompose_arrays`` etc. read ``operator``
    directly per chunk instead) - only ``active_x``'s length
    (``n_active == dim``) matters. Deciding this from an explicit flag
    rather than ``len(x_nz) == dim * dim`` is what makes a genuinely
    empty (zero-cost) placeholder safe to pass here, instead of the
    dim*dim-length ``np.empty`` array (512 MiB at qubits=13) an earlier
    version of this fast path allocated solely so this length check
    would pass - a real, measured, now-fixed memory cost, since a
    dense fast-path caller's peak RSS included that placeholder even
    though nothing ever read it.

    Skipping the redundant sort here measured ~2x faster on this stage
    alone at dim=2048 (dense random Hermitian, qubits=11) and this
    stage was ~53-67% of total `parallel_decompose_arrays` time on
    dense input (see the performance-investigation plan/PLAN.md) -
    output is verified bit-identical to the general path it replaces
    before being trusted (see tests/test_fwht.py).

    For sparse input ``is_fully_dense`` is always False and the general
    ``np.unique`` path runs unchanged - this never touches the
    sparse/structured-input code path's behavior or performance.
    """
    if is_fully_dense:
        return np.arange(dim), x_nz
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    return active_x, inverse


def fwht_pauli_coefficients(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    sparse: bool = False,
    chunk_size: int | None = None,
    atol: float = 1e-10,
    checkpoint_path: str | Path | None = None,
    assume_dense: bool = False,
) -> (
    NDArray[np.complexfloating]
    | tuple[NDArray[np.intp], NDArray[np.complexfloating]]
    | tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.complexfloating]]
):
    """Decompose an operator into Pauli-string coefficients via FWHT.

    Works for **any** complex ``(2**n, 2**n)`` matrix, Hermitian or
    not - see the module docstring's "Hermitian and non-Hermitian
    operators" section. Hermitian input (e.g. a real-symmetric
    coupled-oscillator Hamiltonian) yields real coefficients; general
    input yields complex coefficients.

    Args:
        operator: A ``(2**n, 2**n)`` matrix (real or complex).
            Dimension must be an exact power of two - pad with
            ``paulikit.hamiltonian.pad_to_power_of_two`` first if not.
        sparse: If ``False`` (default), returns the full dense
            ``(dim, dim)`` array described below - unchanged behavior,
            kept as the default so existing callers are unaffected. If
            ``True``, returns only the active (nonzero-row) data
            without ever materializing the dense array - see Returns.
            The dense array's O(dim**2) memory/re-scan cost dwarfs L3
            cache well before it dwarfs the operator's own sparsity
            (confirmed via hardware performance counters);
            ``sparse=True`` is the fix for callers that only need the
            nonzero terms, such as ``fwht_pauli_terms``. Both modes
            compute identically up to the final step - this only
            changes what's returned, not the sparsity-aware
            computation itself.
        chunk_size: Only used when ``sparse=True``. If ``None``
            (default), all active rows are gathered and transformed at
            once (one ``(n_active, dim)`` array live in memory - fine
            for moderate N; returns the dense-block form described
            below). If set to a positive integer, active rows are
            processed in blocks of at most ``chunk_size`` rows - the
            tiling technique from MIT 6.172 lecture 1 (block the
            computation to bound working-set size), applied here for
            memory-footprint reduction rather than cache reuse, since
            each row transforms independently (see
            ``_walsh_hadamard_transform_rows``) with no cross-row
            reduction to preserve. Unlike ``chunk_size=None``, this
            mode also thresholds each chunk's output against ``atol``
            immediately and accumulates only the surviving ``(x, z,
            coefficient)`` triples (see Returns) - bounding peak memory
            to ``O(chunk_size * dim + n_final_terms)`` rather than
            ``O(n_active * dim)``, since an earlier ``chunk_size``
            design still allocated one full ``(n_active, dim)``
            accumulator regardless of ``chunk_size`` - that
            accumulator was the actual N=150 ceiling, not the
            per-chunk transient ``chunk_size`` was designed to bound.
        atol: Only used when ``chunk_size`` is set. Coefficients with
            ``abs(coefficient) <= atol`` are dropped per-chunk, before
            accumulation - moved here (rather than staying purely in
            ``fwht_pauli_terms``) because thresholding must happen
            before accumulation for the space-complexity fix above to
            work; the dense-block modes (``chunk_size=None``) are
            unaffected and keep returning unthresholded output.
        checkpoint_path: Only used when ``chunk_size`` is set. If
            given, each completed chunk's surviving triples are
            appended to ``checkpoint_path`` as one binary chunk-framed
            record - a small fixed-width header (magic, format
            version, index dtype, chunk index, term count) followed by
            the raw ``x``/``z``/``coeff`` arrays' bytes, with no
            per-term Python object construction on the write path -
            and a sibling ``<checkpoint_path>.progress.json`` file
            records progress as one 8-byte little-endian ``u64``
            record per completed chunk, appended in order; the next
            chunk to process is one past the highest recorded index.
            If a
            checkpoint already exists at this path when called, chunks
            already recorded there are skipped and replayed frame by
            frame, one original chunk at a time, rather than
            recomputed - resuming a crashed or interrupted run rather
            than restarting from chunk 0. This costs one small file
            append per chunk (negligible next to each chunk's
            O(chunk_size * dim * log dim) transform cost), so it is
            opt-in but effectively free when enabled; ``None``
            (default) does no I/O at all.

    Returns:
        If ``sparse=False``: a complex ``numpy.ndarray`` of shape
        ``(2**n, 2**n)`` where entry ``[x, z]`` is the coefficient of
        the Pauli string ``P(x, z)`` (see module docstring for the
        x/z encoding). Most entries will be exactly or near zero for
        structured/sparse input; callers that want only the nonzero
        terms should filter by magnitude (see ``fwht_pauli_terms``).

        If ``sparse=True`` and ``chunk_size=None``: a tuple
        ``(active_x, active_coefficients)`` where ``active_x`` is a
        1-D integer array of the ``x`` values with at least one
        nonzero coefficient, and ``active_coefficients`` is a complex
        array of shape ``(len(active_x), dim)`` such that
        ``active_coefficients[i, z]`` is the coefficient of
        ``P(active_x[i], z)``. Rows for ``x`` not in ``active_x`` are
        implicitly all-zero and are not represented at all.

        If ``sparse=True`` and ``chunk_size`` is a positive integer: a
        tuple ``(x_out, z_out, coefficient_out)`` of three 1-D arrays
        of equal length - a COO-style triple where entry ``i`` is the
        coefficient ``coefficient_out[i]`` of Pauli string
        ``P(x_out[i], z_out[i])``, already thresholded against
        ``atol`` (unlike the other two return forms).

    Raises:
        ValueError: If ``operator`` is not square or its dimension is
            not a power of two.

    ``assume_dense`` (default ``False``): same meaning as
    ``parallel_decompose_arrays``'s own parameter of the same name -
    skips the ``np.nonzero`` sparsity-DETECTION scan when ``True`` and
    ``operator`` is a plain ndarray. See that function's docstring for
    the full rationale and the measurement behind it (a pure
    performance choice, not a correctness risk - correct for any
    actual sparsity pattern).
    """
    (
        operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
    ) = _prepare_operator_for_fwht(
        operator, assume_dense=assume_dense
    )
    active_x, inverse = _active_x_and_inverse(
        x_nz, dim, is_fully_dense=assume_dense and not is_sparse_input
    )
    n_active = len(active_x)

    z_indices = np.arange(dim)[np.newaxis, :]

    if sparse and chunk_size is not None:
        # Pre-size the accumulators instead of doubling up from 1024.
        #
        # Each active row contributes at most `dim` terms, so
        # n_active * dim is a hard upper bound - but allocating that
        # outright would defeat the whole point of the chunked path
        # (it is the dense size). Instead seed with a measured-shape
        # estimate and let doubling handle any shortfall.
        #
        # Why it matters: growth, not appending, dominates. Measured
        # at the N=100 shape, extend ran at 3423 MiB/s against a
        # 10162 MiB/s slice-assign ceiling - 66% of its time spent
        # reallocating and copying, across ~17 doublings from 1024 to
        # ~91.6M entries, each copying everything accumulated so far.
        #
        # The estimate: surviving terms measured 0.497, 0.499, 0.500,
        # 0.500 and 0.500 of the n_active * dim bound at N = 20, 30,
        # 50, 100 and 150 - so half the bound is not a guess but a
        # stable structural property of the transform (each active
        # row's WHT spreads its nonzeros across the row, and about
        # half clear atol). One doubling from here still covers the
        # hard bound if an operator ever exceeds it.
        #
        # It also LOWERS peak memory rather than raising it: at N=150
        # this allocates 2.73 GiB once, where doubling up to the same
        # size holds the old and new arrays together at the final
        # reallocation - 4.10 GiB.
        estimated = max(1024, (n_active * dim) // 2)
        x_out = _GrowableArray(np.intp, estimated)
        z_out = _GrowableArray(np.intp, estimated)
        coeff_out = _GrowableArray(complex, estimated)
        for chunk_x_out, chunk_z_out, chunk_coeff_out in _iter_chunked_coefficients(
            operator, is_sparse_input, active_x, inverse, p_nz, q_nz, values_nz,
            dim, n_qubits,
            n_active, z_indices, chunk_size, atol, checkpoint_path,
        ):
            x_out.extend(chunk_x_out)
            z_out.extend(chunk_z_out)
            coeff_out.extend(chunk_coeff_out)

        return x_out.finalize(), z_out.finalize(), coeff_out.finalize()

    gathered_active = np.zeros((n_active, dim), dtype=complex)
    gathered_active[inverse, q_nz] = values_nz

    # Step 2: Walsh-Hadamard Transform of each active row (each fixed
    # x with at least one nonzero gathered entry). Rows with no
    # nonzero entries transform to all-zero and are skipped entirely.
    # overwrite_input=True: gathered_active is never read again after
    # this call, so transforming it in place avoids a second full-size
    # copy (~2.73GiB at N=150) that was otherwise the dominant memory
    # cost in this function - see _walsh_hadamard_transform_rows.
    transformed_active = _walsh_hadamard_transform_rows(
        gathered_active, overwrite_input=True
    )

    # Step 3: phase-factor multiplication, computed only for active x.
    xz_and = active_x[:, np.newaxis] & z_indices
    conj_phase = _phase_from_popcount(xz_and, n_qubits, conjugate=True)
    active_coefficients = transformed_active * (conj_phase * (1.0 / dim))

    if sparse:
        return active_x, active_coefficients

    coefficients = np.zeros((dim, dim), dtype=complex)
    coefficients[active_x] = active_coefficients
    return coefficients


def pauli_label(x_mask: int, z_mask: int, n_qubits: int) -> str:
    """Convert an (x, z) symplectic bitmask pair to an IXYZ label string.

    Per qubit j (bit position ``n_qubits - 1 - j`` of the masks, matching
    this module's row/column convention - see module docstring):
    (x_j, z_j) = (0,0) -> 'I', (1,0) -> 'X', (0,1) -> 'Z', (1,1) -> 'Y'.

    Args:
        x_mask: Integer in [0, 2**n_qubits).
        z_mask: Integer in [0, 2**n_qubits).
        n_qubits: Number of qubits.

    Returns:
        A string of length ``n_qubits``, leftmost character = qubit 0,
        matching the convention used by
        ``paulikit.testing.fixtures.pauli_word_to_label``.
    """
    letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    chars = []
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit
        xj = (x_mask >> bit) & 1
        zj = (z_mask >> bit) & 1
        chars.append(letters[(xj, zj)])
    return "".join(chars)


def _build_real_terms(
    labels: list[str],
    coefficient_values: NDArray[np.complexfloating],
    atol: float,
) -> dict[str, float]:
    """Shared ``assume_hermitian=True`` term-dict builder for
    ``fwht_pauli_terms``/``fwht_pauli_terms_iter``.

    Vectorizes what was previously a per-term Python loop doing an
    ``abs()``/``max()``/comparison Hermiticity check plus a per-item
    dict insert - measured as roughly 60% of total pipeline time at
    N=150, dominated by the per-term check rather
    than the dict construction itself. The tolerance floor here must
    match the original scalar form's ``max(atol, 1e-6 * abs(c))``
    exactly - ``abs(c)`` is the *full complex magnitude*, not
    ``abs(c.real)`` (an easy, non-equivalent substitution: they only
    agree when the imaginary part is already negligible, which is
    exactly the case this check exists to catch).

    On violation, falls back to the same per-term scan the old code
    always ran, only for the (rare) purpose of finding the first
    offending term and reconstructing today's exact error message -
    this keeps the common, non-violating path fully vectorized while
    losing none of the original diagnostic specificity.
    """
    c_abs = np.abs(coefficient_values)
    imag_abs = np.abs(coefficient_values.imag)
    violation = imag_abs > np.maximum(atol, 1e-6 * c_abs)
    if violation.any():
        first = int(np.nonzero(violation)[0][0])
        label = labels[first]
        c = coefficient_values[first]
        raise ValueError(
            f"term {label!r} has non-negligible "
            f"imaginary part {c.imag!r} - operator may not be Hermitian; "
            "pass assume_hermitian=False to decompose it anyway"
        )
    return dict(zip(labels, coefficient_values.real.tolist()))


def _check_hermitian_violation(
    coefficient_values: NDArray[np.complexfloating],
    atol: float,
    x: NDArray[np.integer],
    z: NDArray[np.integer],
    n_qubits: int,
) -> None:
    """Raise if any coefficient has a non-negligible imaginary part.

    The array-yielding path's counterpart to the check inside
    ``_build_real_terms`` - same tolerance rule, same error message,
    but without building a label for every term first. On violation it
    labels ONLY the single offending term, so the diagnostic is
    byte-identical at O(1) cost rather than O(t_i).

    The tolerance floor must match ``_build_real_terms`` exactly:
    ``abs(c)`` is the *full complex magnitude*, not ``abs(c.real)``
    (they only agree when the imaginary part is already negligible,
    which is exactly the case this check exists to catch).

    Compiled fast path: a single fused C pass
    (``hermitian_check.c``) replacing what was previously four
    separate NumPy passes (``np.abs`` twice, ``np.maximum``, the
    comparison-and-``.any()``), each materializing a full-size
    temporary. Isolated measurement (perf stat ``instructions:u``,
    with the test array's own construction cost subtracted out) found
    the fused pass costs as little as ~41% of the 4-pass NumPy
    version. Stops at the first violation, matching this function's
    own "labels only the single offending term" contract - a further,
    input-dependent saving on top of the pass-count reduction.
    """
    if _hermitian_check_native is not None:
        contiguous = np.ascontiguousarray(coefficient_values)
        first_violation = _hermitian_check_native.first_hermitian_violation(
            contiguous, atol
        )
        if first_violation < 0:
            return
        first = first_violation
    else:
        c_abs = np.abs(coefficient_values)
        imag_abs = np.abs(coefficient_values.imag)
        violation = imag_abs > np.maximum(atol, 1e-6 * c_abs)
        if not violation.any():
            return
        first = int(np.nonzero(violation)[0][0])
    label = _pauli_label_batch(x[first:first + 1], z[first:first + 1], n_qubits)[0]
    c = coefficient_values[first]
    raise ValueError(
        f"term {label!r} has non-negligible "
        f"imaginary part {c.imag!r} - operator may not be Hermitian; "
        "pass assume_hermitian=False to decompose it anyway"
    )


def terms_from_arrays(
    x: NDArray[np.integer],
    z: NDArray[np.integer],
    coeff: NDArray[np.complexfloating],
    n_qubits: int,
    assume_hermitian: bool = True,
    atol: float = 1e-10,
) -> dict[str, complex] | dict[str, float]:
    """Render one chunk's ``(x, z, coeff)`` arrays to a label -> coefficient dict.

    The opt-in counterpart to ``parallel_decompose_arrays``: that
    function yields raw arrays so the ~91.6M Python
    ``str`` objects and dict insertions a full decomposition would
    otherwise need are never built in the parent process - which is
    what lifts the multi-core speedup ceiling from ~1.21x to a measured
    ~2.19x. This function is where a caller opts back IN to labels,
    for as many terms as they actually want.

    Building labels for every term of a large decomposition costs the
    same here as it does inside ``parallel_decompose``; the saving
    comes from calling this on a *subset* (filter by coefficient
    magnitude, take the largest terms, render one chunk) rather than
    on all of them.

    Args:
        x: Per-term x bitmasks, any integer dtype (``uint16`` from a
            fresh run, ``intp`` from a legacy checkpoint - both work).
        z: Per-term z bitmasks, same length and dtype rules as ``x``.
        coeff: Per-term complex coefficients.
        n_qubits: Number of qubits, i.e. ``int(log2(dim))``.
        assume_hermitian: If ``True`` (default), raises ``ValueError``
            when any coefficient has a non-negligible imaginary part
            and returns real coefficients - identical contract and
            identical error message to ``fwht_pauli_terms``. If
            ``False``, returns complex coefficients unchecked.
        atol: Tolerance floor for the Hermiticity check.

    Returns:
        ``dict[str, float]`` when ``assume_hermitian=True``, else
        ``dict[str, complex]``.
    """
    labels = _pauli_label_batch(x, z, n_qubits)
    if assume_hermitian:
        return _build_real_terms(labels, coeff, atol)
    return {label: complex(c) for label, c in zip(labels, coeff.tolist())}


def fwht_pauli_terms(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    chunk_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, complex] | dict[str, float]:
    """Decompose an operator into a label -> coefficient dict.

    Convenience wrapper around ``fwht_pauli_coefficients`` that filters
    to nonzero terms and converts (x, z) bitmask pairs to IXYZ label
    strings, matching the format used by
    ``paulikit.testing.fixtures``.

    Works for **any** complex ``(2**n, 2**n)`` matrix, not just
    Hermitian operators: the Pauli strings (including Y) span the
    full space of complex matrices, so an arbitrary (non-Hermitian,
    non-normal, non-diagonalizable - anything) linear operator has a
    well-defined Pauli decomposition with, in general, complex
    coefficients. This matters beyond mathematical completeness:
    physically, not every operator of interest is Hermitian - e.g.
    non-Hermitian effective Hamiltonians for open/dissipative quantum
    systems (complex energies encode finite lifetimes), PT-symmetric
    Hamiltonians, Liouvillian superoperators, or terms in a sum where
    only the total is Hermitian even though individual summands are
    not.

    Args:
        operator: A ``(2**n, 2**n)`` complex or real matrix. No
            Hermiticity is assumed unless ``assume_hermitian=True``.
        atol: Terms with ``abs(coefficient) <= atol`` are dropped.
        assume_hermitian: If True, asserts every returned
            coefficient's imaginary part is within ``atol`` of zero
            (raising ``ValueError`` otherwise) and returns real
            (``float``) coefficients. If False (the default), returns
            ``complex`` coefficients as-is - correct for both
            Hermitian and non-Hermitian input, since a Hermitian
            operator's Pauli coefficients are real to begin with (the
            imaginary parts are then just zero, not dropped).
        chunk_size: Passed through to ``fwht_pauli_coefficients`` - see
            its docstring. ``None`` (default) processes all active
            rows at once, then thresholds by ``atol`` here. A positive
            integer switches to the chunked, already-thresholded COO
            path (``atol`` is applied per-chunk inside
            ``fwht_pauli_coefficients`` in that case, not here - same
            numerical result, but memory stays bounded at large N.
        checkpoint_path: Only meaningful when ``chunk_size`` is set -
            passed through to ``fwht_pauli_coefficients``, see its
            docstring for the resume behavior.

    Returns:
        A dict mapping Pauli-string label (e.g. ``"IXZ"``) to its
        coefficient: ``float`` if ``assume_hermitian=True``,
        ``complex`` otherwise.

    Raises:
        ValueError: If ``assume_hermitian=True`` and a term's
            coefficient has a non-negligible imaginary part.
    """
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))

    if chunk_size is not None:
        # Already-thresholded COO triples - see fwht_pauli_coefficients's
        # chunk_size/atol docstring. No further thresholding/gather
        # needed here; that is the whole point of this path.
        x_nonzero, z_nonzero, coefficient_values = fwht_pauli_coefficients(
            operator,
            sparse=True,
            chunk_size=chunk_size,
            atol=atol,
            checkpoint_path=checkpoint_path,
        )
    else:
        active_x, active_coefficients = fwht_pauli_coefficients(operator, sparse=True)
        # Re-scan only the active rows fwht_pauli_coefficients already
        # identified, never the full (dim, dim) array - the dense
        # re-scan was a measured cache-locality/robustness problem
        # (OOMs at N=150).
        row_idx, z_nonzero = np.nonzero(np.abs(active_coefficients) > atol)
        x_nonzero = active_x[row_idx]
        coefficient_values = active_coefficients[row_idx, z_nonzero]

    labels = _pauli_label_batch(x_nonzero, z_nonzero, n_qubits)

    if assume_hermitian:
        return _build_real_terms(labels, coefficient_values, atol)

    complex_terms: dict[str, complex] = {
        label: complex(c) for label, c in zip(labels, coefficient_values.tolist())
    }
    return complex_terms


def fwht_pauli_terms_iter(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    chunk_size: int,
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    checkpoint_path: str | Path | None = None,
    parallel_labels: bool = False,
    assume_dense: bool = False,
    chunk_plan: dict | None = None,
) -> Iterator[dict[str, complex] | dict[str, float]]:
    """Streaming counterpart to ``fwht_pauli_terms``. Yields one
    ``dict`` of terms per chunk instead of building one combined
    ``dict`` for the whole operator.

    Why this exists (the actual problem, not just a memory
    workaround): each chunk of active rows is a fully independent
    sub-problem - there is no cross-chunk combination step anywhere in
    the underlying math (unlike, say, tiled matrix multiply's
    block-sum reduction). ``fwht_pauli_terms`` re-fuses every chunk's
    result into one dict before the caller ever sees it, which is an
    artificial recombination the math does not require, not a
    necessary step - the actual divide-and-conquer strategy for this
    problem is to keep each tile a tile all the way to the caller.
    This matters beyond just fitting in memory: at N=150, the full
    combined result is 91.65M terms (several GiB for the raw data
    alone, more once label strings and a single dict's hash-table
    overhead are added), too large for a dict-returning API to handle
    on modest hardware, regardless of any per-chunk memory bound. A
    caller that only needs to, e.g.,
    write terms to disk, filter them, or fold them into a running sum
    never needs the full combined dict to exist at once.

    Args:
        operator: Same contract as ``fwht_pauli_terms``.
        chunk_size: Required (no default) - unlike
            ``fwht_pauli_terms``, there is no dense/whole-array mode
            here; streaming without chunking is not a meaningful
            combination.
        atol: Same as ``fwht_pauli_terms`` - applied per-chunk inside
            ``fwht_pauli_coefficients``.
        assume_hermitian: Same as ``fwht_pauli_terms``, but checked
            per-chunk rather than for the whole operator at once: a
            ``ValueError`` raised on chunk *k* means chunks
            ``0..k-1`` were already yielded to the caller before the
            error. This is a real behavior difference from
            ``fwht_pauli_terms``, which
            either returns a fully valid dict or raises before
            returning anything; a streaming caller that needs
            all-or-nothing Hermiticity validation should call
            ``fwht_pauli_terms`` (non-streaming) instead.
        checkpoint_path: Same as ``fwht_pauli_terms`` - passed through
            to ``fwht_pauli_coefficients``'s chunked accumulation
            internals for crash/resume. Checkpoints are stored as a
            binary chunk-framed format (one
            self-describing frame per completed chunk - see
            ``_append_checkpoint_frame``/``_iter_checkpoint_frames``),
            not a per-term text log. Note this checkpoints the
            underlying coefficient computation, not this generator's
            own iteration state - resuming a streaming consumer that
            was itself interrupted partway through consuming chunks
            means simply calling this function again with the same
            ``checkpoint_path``; already checkpointed chunks are
            replayed one frame at a time, per original chunk (see
            ``_iter_chunked_coefficients``), so a consumer sees the
            same chunk boundaries a resumed run would have produced
            on a first pass, not one merged replay tile.
        parallel_labels: If ``True``, uses the oneTBB-parallel label
            kernel (``pauli_label_batch_parallel``) per chunk instead
            of the serial kernel. Measured **in isolation** as a real
            ~1.1-1.4x wall-clock win at N=150-representative scale, at
            the cost of a modest cache-locality regression. However,
            re-measured embedded in the real streaming pipeline at
            N=150, this delivers **no measurable wall-clock or
            cache-locality difference either way** - label generation
            is only ~7% of
            total pipeline time, dwarfed by dict construction (~60%),
            so the isolated effect washes out to noise at the
            whole-pipeline level. Left opt-in (default ``False``) since
            it is not harmful, just not a meaningful lever for this
            pipeline's actual performance.

    Yields:
        One ``dict`` per chunk, same value-type contract as
        ``fwht_pauli_terms`` (``float`` values if
        ``assume_hermitian=True``, ``complex`` otherwise). A chunk
        with no surviving terms above ``atol`` yields an empty dict,
        not a skipped chunk - callers that want to skip empty chunks
        should filter for truthiness themselves.

    Raises:
        ValueError: If ``operator`` is not square or its dimension is
            not a power of two (raised immediately, before the first
            chunk is yielded - this check does not depend on any
            chunk's data). Also raised mid-stream, after already
            yielding zero or more prior chunks, if
            ``assume_hermitian=True`` and a term in the current chunk
            has a non-negligible imaginary part - see the
            ``assume_hermitian`` parameter above.

    ``assume_dense`` (default ``False``): same meaning as
    ``parallel_decompose_arrays``'s own parameter of the same name -
    see that function's docstring for the full rationale.
    """
    (
        operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
    ) = _prepare_operator_for_fwht(
        operator, assume_dense=assume_dense
    )
    active_x, inverse = _active_x_and_inverse(
        x_nz, dim, is_fully_dense=assume_dense and not is_sparse_input
    )
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]

    for chunk_x, chunk_z, chunk_coeff in _iter_chunked_coefficients(
        operator, is_sparse_input, active_x, inverse, p_nz, q_nz, values_nz,
        dim, n_qubits,
        n_active, z_indices, chunk_size, atol, checkpoint_path,
        chunk_plan=chunk_plan,
    ):
        labels = _pauli_label_batch(chunk_x, chunk_z, n_qubits, parallel=parallel_labels)

        if assume_hermitian:
            yield _build_real_terms(labels, chunk_coeff, atol)
        else:
            yield {
                label: complex(c) for label, c in zip(labels, chunk_coeff.tolist())
            }


# Multiplier applied to the naive dim**2*16 (one (dim, dim) complex128
# array) estimate to approximate the dense path's REAL peak footprint.
# The naive estimate only accounts for gathered_active/
# transformed_active - it misses several other same-order-of-magnitude
# arrays concurrently live during fwht_pauli_coefficients/
# fwht_pauli_terms's dense (sparse=True, chunk_size=None) path: xz_and
# (int64, dim**2*8), _popcount_array's uint32 cast and int64 count
# accumulator (dim**2*4 + dim**2*8), phase (complex128, dim**2*16),
# active_coefficients (complex128, dim**2*16), the
# np.abs(active_coefficients) boolean-mask intermediate
# (float64, dim**2*8) in fwht_pauli_terms's own nonzero-rescan, plus
# label-string and dict-construction overhead after that. A real
# resource.getrusage(RUSAGE_SELF).ru_maxrss sweep at N=50/75/100
# (follow-up fix, 2026-09-01) measured this ratio directly: 5.63x,
# 6.47x, 5.27x respectively (N=25's 18.20x is a small-N artifact -
# fixed Python/NumPy process baseline RSS dominates at that scale, not
# representative) - consistently in the 5-6.5x range, clustering
# around 6x. At N=150 the true ratio is higher still: even an 18 GiB
# cap (4.5x the naive 4.00 GiB estimate) was insufficient (re-verified
# during this fix) - 6x alone would NOT have been safe at N=150
# without also tightening _DENSE_MEMORY_SAFETY_FRACTION below.
_DENSE_MEMORY_MULTIPLIER = 6.0

# Fraction of the available memory budget (autotune.available_memory_bytes)
# the dense path's estimated peak footprint (already inflated by
# _DENSE_MEMORY_MULTIPLIER above) must stay under to be chosen - leaves
# further headroom for the operator array itself, Python/NumPy
# overhead, other processes on a shared node, and the real ratio being
# somewhat higher than the 5-6.5x measured range (N=150's own
# real-world ratio was not fully bounded above - measurement stopped
# once even an 18 GiB cap failed, see _DENSE_MEMORY_MULTIPLIER's own
# comment). Deliberately small (0.2, not 0.5) after the previous
# 0.5/naive-estimate combination was measured to underestimate real
# peak usage by 3x+ in the unsafe direction at N=150.
_DENSE_MEMORY_SAFETY_FRACTION = 0.2


def auto_decompose(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    checkpoint_path: str | Path | None = None,
) -> dict[str, complex] | dict[str, float] | Iterator[dict[str, complex] | dict[str, float]]:
    """Auto-picks streaming vs. dense and an auto-tuned ``chunk_size``.
    Returns either a ``dict`` (dense path, same as
    calling ``fwht_pauli_terms`` with no ``chunk_size``) or an
    ``Iterator[dict]`` (streaming path, same as calling
    ``fwht_pauli_terms_iter``) depending on a runtime decision based on
    the operator's size and the machine's currently-available memory.

    **This return type is a runtime decision, not a fixed contract** -
    unlike ``fwht_pauli_terms``/``fwht_pauli_terms_iter``, which always
    return a ``dict``/``Iterator[dict]`` respectively regardless of
    machine state. This is deliberate: making an *existing* function's
    return type depend on runtime memory state would be a hidden-
    nondeterminism hazard (the same call could silently take a
    different code path on a re-run) and would silently change
    ``assume_hermitian``'s validation-contract difference between the
    two paths (all-or-nothing vs. partial-yield-then-error) out from
    under a caller who never asked for that.
    ``auto_decompose``'s name documents this nondeterminism
    explicitly; callers that need a fixed contract
    should call ``fwht_pauli_terms``/``fwht_pauli_terms_iter`` directly
    instead. A typical caller checks the result with
    ``isinstance(result, dict)``.

    The streaming-vs-dense decision is based on a memory budget
    (``paulikit.algorithms.autotune.available_memory_bytes``) that is
    cgroup-aware, not just physical-RAM-aware - correctness-critical on
    a shared HPC node, where a scheduler (Slurm/PBS) commonly caps a
    job below the node's full physical RAM via a cgroup; using
    physical memory alone there could wrongly choose the dense path
    inside a job actually capped well below what dense would need. The
    streaming path's own ``chunk_size`` is likewise auto-tuned
    (``autotune.recommended_chunk_size``) via an empirical cache-
    latency probe rather than a fixed example value.

    The dense-path memory estimate deliberately errs conservative: a
    real ``resource.getrusage`` sweep found the dense path's actual
    peak footprint fits well under the available budget at some sizes
    where this function's own (already 6x-inflated, see
    ``_DENSE_MEMORY_MULTIPLIER``) estimate says to stream instead.
    This means ``auto_decompose`` will sometimes choose streaming where
    dense would in fact have fit and been somewhat faster; this
    imprecision is intentional, not a bug - for a safety-critical
    memory decision, occasionally streaming when dense would have
    worked is a far better failure mode than occasionally choosing
    dense and running the process out of memory. A caller that knows
    its own memory headroom precisely and wants the dense path's
    typically-faster performance can call ``fwht_pauli_terms`` directly
    instead of going through this estimate.

    Args:
        operator: Same contract as ``fwht_pauli_terms``.
        atol: Same as ``fwht_pauli_terms``.
        assume_hermitian: Same as ``fwht_pauli_terms`` (dense path) /
            ``fwht_pauli_terms_iter`` (streaming path) - see those
            functions' own docstrings for the validation-contract
            difference between them, which still applies here
            depending on which path is chosen.
        checkpoint_path: Same as ``fwht_pauli_terms``/
            ``fwht_pauli_terms_iter`` - only meaningful if the
            streaming path is chosen.

    Returns:
        A ``dict`` if the dense path was chosen, or an
        ``Iterator[dict]`` if the streaming path was chosen - check
        with ``isinstance(result, dict)``.
    """
    from paulikit.algorithms import autotune

    dim = operator.shape[0]
    # Worst-case (fully dense operator) estimated peak footprint of
    # the dense path's accumulator: n_active <= dim active rows, each
    # dim complex128 entries (16 bytes) - matches
    # fwht_pauli_coefficients's own O(n_active * dim) accounting for
    # just its own gathered_active/transformed_active array.
    # Deliberately does not pre-scan the operator to find the real
    # n_active first (that would cost an extra full pass) - this is a
    # cheap, safe upper bound on n_active, not a precise estimate.
    #
    # Multiplied by _DENSE_MEMORY_MULTIPLIER (see its own comment for
    # the real-measurement basis) since the naive dim**2*16 figure
    # alone was measured to underestimate the dense path's REAL peak
    # memory usage by 3x+ at N=150 (several other same-order-of-
    # magnitude intermediate arrays are concurrently live - xz_and,
    # _popcount_array's temporaries, phase, active_coefficients, the
    # nonzero-rescan's boolean mask, label/dict construction).
    estimated_dense_bytes = dim * dim * 16 * _DENSE_MEMORY_MULTIPLIER

    budget = autotune.available_memory_bytes()
    if estimated_dense_bytes <= budget * _DENSE_MEMORY_SAFETY_FRACTION:
        return fwht_pauli_terms(
            operator,
            atol=atol,
            assume_hermitian=assume_hermitian,
            checkpoint_path=checkpoint_path,
        )

    chunk_size = autotune.recommended_chunk_size(dim)
    return fwht_pauli_terms_iter(
        operator,
        chunk_size=chunk_size,
        atol=atol,
        assume_hermitian=assume_hermitian,
        checkpoint_path=checkpoint_path,
    )


# Multi-core chunk parallelism: per-worker process state, set once
# via ProcessPoolExecutor's initializer rather than pickled into
# every task - the operator (and the shared
# sorted/active-x arrays every chunk gathers a slice of) can be
# multiple GiB at real N, so shipping it once per *worker process*
# rather than once per *chunk* is not an optimization here, it is the
# difference between this being usable at all and every task paying an
# O(operator size) pickling cost that dwarfs the chunk's own O(chunk_
# size * dim) work.
_parallel_worker_state: dict | None = None


def _parallel_worker_init(
    operator,
    is_sparse_input: bool,
    sorted_inverse: NDArray[np.intp],
    sorted_p_nz: NDArray[np.intp],
    sorted_q_nz: NDArray[np.intp],
    sorted_values: NDArray[np.complexfloating],
    active_x: NDArray[np.intp],
    dim: int,
    n_qubits: int,
    z_indices: NDArray[np.intp],
    atol: float,
    pin_cpus: list[int] | None,
    next_pin_index,
) -> None:
    """``ProcessPoolExecutor`` initializer - runs once per worker
    process, stashing everything ``_parallel_worker_chunk`` needs in
    that process's own global state so per-task calls only need to
    pass the (tiny) chunk boundaries.

    ``pin_cpus``/``next_pin_index`` implement the CPU-pinning fix:
    each worker process atomically claims the next unused
    index into ``pin_cpus`` (one representative logical CPU per
    PHYSICAL core - see ``_physical_core_representative_cpus``) via
    ``next_pin_index`` (a ``multiprocessing.Value`` shared counter,
    the only way to hand each of several otherwise-identical
    ``initializer`` calls a distinct index - ``ProcessPoolExecutor``
    does not pass a per-worker ordinal itself), then pins itself to
    that one CPU. If ``pin_cpus`` is ``None`` (non-Linux, or fewer
    distinct physical cores than requested workers - see
    ``parallel_decompose``) or more workers claim an index than
    ``pin_cpus`` has entries (can happen if the pool starts more
    worker processes than ``max_workers`` transiently, e.g. during
    ``max_tasks_per_child`` recycling - not used here, but defensive
    regardless), pinning is skipped for the excess worker(s) - a
    worker that isn't pinned is still correct, just not guaranteed
    isolated from a hyperthread sibling.
    """
    global _parallel_worker_state
    _parallel_worker_state = {
        "operator": operator,
        "is_sparse_input": is_sparse_input,
        "sorted_inverse": sorted_inverse,
        "sorted_p_nz": sorted_p_nz,
        "sorted_q_nz": sorted_q_nz,
        "sorted_values": sorted_values,
        "active_x": active_x,
        "dim": dim,
        "n_qubits": n_qubits,
        "z_indices": z_indices,
        "atol": atol,
    }

    if pin_cpus:
        with next_pin_index.get_lock():
            my_index = next_pin_index.value
            next_pin_index.value += 1
        if my_index < len(pin_cpus):
            _pin_current_process_to_cpu(pin_cpus[my_index])


def _index_dtype_for_dim(dim: int) -> np.dtype:
    """Smallest unsigned dtype that can hold any (x, z) index for a
    ``dim``-wide problem - reduces IPC payload size.

    Both index arrays returned by ``_parallel_worker_chunk`` are
    bounded by ``dim``: ``z_idx`` is a column index into a
    ``(chunk_size, dim)`` array, and ``chunk_x_out`` holds ``active_x``
    *values*, which are row XOR bitmasks over ``n_qubits`` bits and so
    are also strictly less than ``dim = 2**n_qubits``. NumPy defaults
    both to ``intp`` (8 bytes on this platform), which is 4x wider than
    needed at the real N=150 workload and is paid on every one of the
    thousands of chunks that cross the process boundary.

    The dtype is chosen FROM ``dim`` rather than hardcoded, which
    matters for correctness, not just size: ``uint16`` holds indices
    only while ``dim <= 65536`` (``n_qubits <= 16``). Hardcoding it
    would silently WRAP for a 17-qubit or larger operator, producing
    wrong Pauli labels rather than an error. Falls back to ``intp``
    above ``uint32`` range so the function is total for any input.
    """
    if dim <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if dim <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    return np.dtype(np.intp)


def _parallel_worker_chunk(
    chunk_index: int, chunk_start: int, chunk_end: int
) -> tuple[int, NDArray[np.unsignedinteger], NDArray[np.unsignedinteger], NDArray[np.complexfloating]]:
    """Runs in a worker process (via the pool started by
    ``parallel_decompose``): computes exactly one chunk's ``(x, z,
    coefficient)`` triples - the same per-chunk body as
    ``_iter_chunked_coefficients``, factored out so it can run as an
    independent task with no generator/closure state to pickle.

    Returns ``(chunk_index, chunk_x_out, z_idx, chunk_coeff_out)`` -
    the index is threaded through so the main process can checkpoint
    and reassemble results regardless of which order the pool's
    ``as_completed`` delivers them in (workers do not complete chunks
    in submission order).

    NOTE (reverts an earlier design that moved labeling into the
    worker): labeling (``_pauli_label_batch``) was tried here instead of in
    ``parallel_decompose``'s drain loop, on the theory that it would
    let that Theta(t_i * n_qubits) cost run in parallel instead of
    being serialized once per chunk. Measured real regression instead
    (~18-23% slower wall-clock across every worker/core configuration,
    including n_workers=1, where nothing about parallelism should have
    changed at all): returning ``labels`` (a ``list[str]``) through the
    ProcessPoolExecutor's result queue costs ~4x more to pickle than
    the raw NumPy arrays alone (measured ~1.76ms/chunk extra, ~9.8s
    total at N=150's 5595 chunks) - MORE than the ~5s total the
    labeling compute itself costs even fully serialized. The DAG
    analysis that motivated the change was correct about the
    computation graph it modeled, but that graph has no node at all
    for IPC/pickling cost - a real, now-documented gap in the
    methodology, not an arithmetic error in the Work/Span numbers it
    did compute. Reverted based on that measurement.
    """
    state = _parallel_worker_state
    assert state is not None, "_parallel_worker_init must run before _parallel_worker_chunk"

    dim = state["dim"]
    active_x = state["active_x"]

    # FULLY DENSE fast path: skip the global sort-and-searchsorted
    # gather entirely. When len(active_x) == dim, active_x is provably
    # arange(dim) (see _active_x_and_inverse) and EVERY row/column of
    # `operator` is nonzero, so gathered_chunk[row, q] = operator[p, q]
    # for p = x ^ q (XOR is its own inverse) can be read directly via
    # one fancy-index gather, with no dependency on the sorted_*
    # arrays or the chunk's position within them at all. This replaces
    # what was measured (on real dense random-Hermitian input) to be
    # ~40-57% of total parallel_decompose_arrays time (the global
    # argsort + per-chunk np.searchsorted + np.zeros-and-scatter) with
    # a single gather whose cost scales with this chunk's own size,
    # not the whole operator - verified bit-identical to the general
    # path's output before being trusted (tests/test_fwht.py,
    # verification/exhaustive_projection.py). Gated on
    # `not is_sparse_input` too (not just len(active_x) == dim) so a
    # scipy.sparse operator that happens to have every cell nonzero
    # still uses CSR fancy indexing via the general path below, rather
    # than this path's dense-ndarray-shaped indexing assumption.
    if not state["is_sparse_input"] and len(active_x) == dim:
        if _gather_native is not None:
            # Compiled fast path: a tight C loop over the underlying
            # doubles (gather.c) replacing NumPy's generic advanced-
            # indexing machinery for this exact XOR-permutation read
            # pattern. Measured a modest but real win (perf stat
            # instructions:u, ~5-7% fewer instructions than the NumPy
            # fallback below at dim=1024) - this stage is memory-bound
            # (an unavoidable dim*dim scattered read, not per-call
            # dispatch overhead), so a C kernel helps less here than it
            # did for the WHT butterfly or coefficient emission, which
            # were bounded by NumPy's per-stage Python overhead rather
            # than the underlying memory traffic.
            gathered_chunk = _gather_native.gather_dense_chunk(
                state["operator"], chunk_start, chunk_end - chunk_start
            )
        else:
            x_values = np.arange(chunk_start, chunk_end)
            q_range = state["z_indices"][0]
            p_indices = x_values[:, np.newaxis] ^ q_range[np.newaxis, :]
            gathered_chunk = np.ascontiguousarray(
                state["operator"][p_indices, q_range[np.newaxis, :]]
            )
    else:
        sorted_inverse = state["sorted_inverse"]
        lo = int(np.searchsorted(sorted_inverse, chunk_start))
        hi = int(np.searchsorted(sorted_inverse, chunk_end))

        gathered_chunk = np.zeros((chunk_end - chunk_start, dim), dtype=complex)
        # A slice of the values extracted once in the parent, not a per
        # chunk operator lookup - see _iter_chunked_coefficients. This
        # runs in every worker on every chunk, so it is the hottest
        # instance of the 45.4us-per-chunk scipy overhead.
        gathered_values = state["sorted_values"][lo:hi]
        gathered_chunk[
            sorted_inverse[lo:hi] - chunk_start, state["sorted_q_nz"][lo:hi]
        ] = gathered_values

    transformed_chunk = _walsh_hadamard_transform_rows(gathered_chunk, overwrite_input=True)

    chunk_x_out, z_idx, chunk_coeff_out = _coefficients_from_transformed(
        transformed_chunk, active_x[chunk_start:chunk_end],
        state["z_indices"], state["n_qubits"], 1.0 / dim, state["atol"],
    )

    # Narrow ONLY the two index arrays before they cross the process
    # boundary (see _index_dtype_for_dim). The NumPy fallback path
    # returns both as intp (8 bytes); at dim=16384 two bytes suffice,
    # cutting the pickled per-chunk payload measurably with no loss of
    # information - every value is provably < dim, so the cast is
    # exact, not lossy. When the compiled kernel supplied these arrays
    # they are already at this width and the cast is a no-op (it is
    # left in place rather than made conditional: it is free when
    # unnecessary and load-bearing for the fallback).
    #
    # chunk_coeff_out is deliberately NOT narrowed to its real part
    # here, even though assume_hermitian=True callers only use the real
    # part in the end: _build_real_terms performs the Hermiticity
    # violation check ON THE IMAGINARY PART (fwht.py's
    # `imag_abs > np.maximum(atol, 1e-6 * c_abs)`), and that check runs
    # in the parent AFTER this value crosses IPC. Sending only the real
    # part would not shrink a payload so much as silently delete the
    # evidence that check exists to find, turning a raised ValueError
    # into a wrong answer for a non-Hermitian operator.
    idx_dtype = _index_dtype_for_dim(dim)
    return (
        chunk_index,
        chunk_x_out.astype(idx_dtype, copy=False),
        z_idx.astype(idx_dtype, copy=False),
        chunk_coeff_out,
    )


def _detect_available_worker_count() -> int:
    """Number of CPUs actually usable by *this process* right now -
    a correctness fix versus the naive
    ``os.cpu_count()``/``multiprocessing.cpu_count()``, both of which
    report a node's *total* core count even inside a cgroup/cpuset-
    restricted HPC job (the identical bug class already fixed for
    memory - see ``autotune.available_memory_bytes`` versus raw
    ``/proc/meminfo`` ``MemTotal``). ``os.sched_getaffinity`` is
    Linux-only; falls back to ``os.cpu_count()`` elsewhere (macOS/BSD),
    a real, documented portability gap.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _physical_core_representative_cpus() -> list[int] | None:
    """One logical CPU id per PHYSICAL core, among the CPUs this
    process is actually allowed to use - fixes a real gap found by
    direct measurement: without explicit pinning,
    ``ProcessPoolExecutor`` workers are freely migrated by
    the Linux scheduler across ALL logical CPUs regardless of
    ``n_workers``, including both hyperthread siblings of the same
    physical core running workers simultaneously - confirmed via
    direct ``ps -o psr`` sampling, not assumed. ``len(os.sched_
    getaffinity(0))`` (``_detect_available_worker_count``) counts
    logical CPUs, which over-counts on a hyperthreaded machine (this
    dev machine: 8 logical CPUs, 4 physical cores) - a distinct
    correctness question from that function's own cgroup/cpuset
    concern.

    Reads ``/sys/devices/system/cpu/cpu<N>/topology/
    thread_siblings_list`` (Linux only) for each CPU this process is
    allowed to use, groups CPUs into physical-core sibling sets, and
    returns one representative CPU id per SET (the lowest id in each
    group) - deterministic and stable across calls. Returns ``None``
    if unavailable (non-Linux, sysfs not mounted, or any read fails) -
    callers must fall back to their own default when this returns
    ``None``.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:
        return None

    core_of: dict[int, int] = {}
    for cpu in sorted(allowed):
        siblings_path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(siblings_path) as f:
                siblings_str = f.read().strip()
        except OSError:
            return None
        # Format: comma-separated list, or ranges like "0-1" - this
        # machine's format ("0,4") is comma-separated; handle a range
        # entry defensively since the sysfs format is not guaranteed
        # identical across kernels.
        siblings: set[int] = set()
        for part in siblings_str.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                siblings.update(range(int(lo), int(hi) + 1))
            elif part:
                siblings.add(int(part))
        physical_core_id = min(siblings) if siblings else cpu
        core_of.setdefault(physical_core_id, cpu)

    return sorted(core_of.values())


def _pin_current_process_to_cpu(cpu: int) -> bool:
    """Pins the CALLING process to a single logical CPU - Linux only
    (``sched_setaffinity`` has no portable POSIX equivalent, same
    caveat as ``cache_probe.c``'s own ``pin_to_one_cpu``). Returns
    ``True`` on success, ``False`` if unavailable/failed (caller
    should treat this as best-effort, not fatal - an unpinned worker
    is still correct, just potentially slower/more contended).
    """
    try:
        os.sched_setaffinity(0, {cpu})
        return True
    except (AttributeError, OSError):
        return False


def _per_worker_resident_bytes(
    operator, is_sparse_input: bool, nnz: int
) -> int:
    """Estimated fixed footprint EVERY ``parallel_decompose`` worker
    holds resident for its whole lifetime, independent of
    ``chunk_size`` - a real gap found by review (REVIEW_NOTES.md
    2026-09-04): ``_recommended_parallel_chunk_size`` bounded only the
    per-chunk transient buffer against the per-worker memory budget,
    never subtracting this fixed cost first, so the memory-bound
    chunk_size was an overestimate on operators large/dense enough for
    this fixed cost to matter.

    Each worker's ``_parallel_worker_init`` (see its own docstring's
    note on pickling cost) receives and holds one full copy of
    ``operator`` plus the ``nnz``-length ``sorted_inverse``/
    ``sorted_p_nz``/``sorted_q_nz`` setup arrays - this estimates that
    total in bytes. Dense operators use ``operator.nbytes`` directly;
    sparse (CSR) operators are estimated via their own
    ``data``/``indices``/``indptr`` buffers rather than a
    dim-squared-equivalent, since a CSR matrix's real resident size is
    ``O(nnz)``, not ``O(dim**2)`` (the whole reason the sparse input
    path exists - see ``_prepare_operator_for_fwht``).
    """
    if is_sparse_input:
        operator_bytes = (
            operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes
        )
    else:
        operator_bytes = operator.nbytes
    # sorted_inverse/sorted_p_nz/sorted_q_nz: three intp arrays of
    # length nnz (see parallel_decompose's own sort-then-slice setup).
    setup_arrays_bytes = 3 * nnz * np.dtype(np.intp).itemsize
    return operator_bytes + setup_arrays_bytes


def _resolve_n_workers(n_workers: int | None) -> int:
    """Shared by ``parallel_decompose`` and ``parallel_decompose_arrays``
    (previously duplicated verbatim in both, a real maintenance risk
    found by code review - any future fix to this heuristic would
    otherwise need applying in two places or silently drift).

    Returns ``n_workers`` unchanged if given explicitly; otherwise
    caps the auto-detected default to the number of distinct PHYSICAL
    cores this process may use, not logical CPUs -
    ``_detect_available_worker_count()`` counts logical CPUs, correct
    for cgroup/cpuset restrictions but over-counting real parallel
    capacity for this CPU-bound workload on a hyperthreaded machine.
    Real measurement found n_workers=2 beats both 4 (physical core
    count on the 4-core/8-thread dev machine) and 8 (logical CPU
    count) on wall-clock, and that neither n_workers=4 nor
    n_workers=8 achieves meaningful isolation without explicit pinning
    (see ``_physical_core_representative_cpus``) - capping to physical
    cores is the evidence-based choice here, not a guess. Falls back
    to the logical-CPU count if the physical-core probe itself is
    unavailable (non-Linux).
    """
    if n_workers is not None:
        return n_workers
    logical_default = _detect_available_worker_count()
    physical_cpus = _physical_core_representative_cpus()
    return len(physical_cpus) if physical_cpus else logical_default


def _recommended_parallel_chunk_size(
    dim: int, n_workers: int, fixed_resident_bytes: int = 0
) -> int:
    """Auto chunk_size for ``parallel_decompose``, accounting for
    memory in a way ``autotune.recommended_chunk_size`` alone does
    not - a bug fix found by the user noticing real memory spikes
    versus the non-parallel chunked path the same day this was first
    shipped.

    ``autotune.recommended_chunk_size(dim)`` only targets cache
    locality for a SINGLE process - it was never memory-bounded
    even in the single-process path (memory there is
    bounded by ``auto_decompose``'s separate streaming-vs-dense
    decision, not by ``chunk_size`` itself). Under parallelism, up to
    ``n_workers`` chunks' ``O(chunk_size * dim)`` working sets are live
    SIMULTANEOUSLY rather than one at a time - reusing the
    single-process cache-driven value unchanged here would multiply
    real peak memory by roughly ``n_workers`` with no corresponding
    check. Returns the smaller of the cache-driven value and the
    largest chunk_size whose working set fits within one worker's
    share of the memory budget
    (``autotune.per_worker_memory_budget_bytes(n_workers)``), after
    first subtracting ``fixed_resident_bytes`` - the operator copy and
    setup arrays every worker holds resident regardless of
    ``chunk_size`` (see ``_per_worker_resident_bytes``) - from that
    per-worker budget, so the remaining chunk_size bound reflects what
    is actually still available for the per-chunk transient buffer.
    """
    from paulikit.algorithms import autotune

    cache_chunk_size = autotune.recommended_chunk_size(dim)
    worker_budget = autotune.per_worker_memory_budget_bytes(n_workers)
    remaining_budget = max(0, worker_budget - fixed_resident_bytes)
    bytes_per_row = dim * 16  # complex128, matches recommended_chunk_size's own accounting
    memory_bound_chunk_size = max(1, remaining_budget // max(bytes_per_row, 1))
    return min(cache_chunk_size, memory_bound_chunk_size)


def parallel_decompose(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    chunk_size: int | None = None,
    n_workers: int | None = None,
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    checkpoint_path: str | Path | None = None,
) -> Iterator[dict[str, complex] | dict[str, float]]:
    """Multi-core counterpart to ``fwht_pauli_terms_iter``.
    Distributes chunks across a ``ProcessPoolExecutor``
    instead of processing them one at a time in this process; each
    chunk is a fully independent sub-problem (no cross-chunk
    combination step in the underlying math - see
    ``_iter_chunked_coefficients``'s own docstring), which is exactly
    what makes this a real, not just nominal, parallelization.

    A new top-level function rather than a parameter added to
    ``fwht_pauli_terms_iter`` - deliberately, matching
    ``auto_decompose``'s own precedent: changing an
    *existing* function's iteration-order/resource-lifetime contract
    based on a new parameter is a bigger compatibility hazard than
    adding a new function with its own, clearly different contract
    (results here are NOT guaranteed to arrive in chunk order - see
    Yields below).

    **The two auto-tuning quantities this depends on
    (``recommended_chunk_size``, ``available_memory_bytes``) were
    measured/derived on a single lone process - see
    Args below for how this function adapts each for real multi-worker
    use rather than reusing them unchanged.**

    Args:
        operator: Same contract as ``fwht_pauli_terms``.
        chunk_size: If ``None`` (default), the smaller of (a)
            ``autotune.recommended_chunk_size(dim)`` (the
            cache-locality formula, derived for a single lone
            process's cache - concurrent workers competing for one
            shared LLC/memory bandwidth may have a different real
            optimum, not yet re-measured under concurrent load) and
            (b) the largest
            chunk_size whose ``O(chunk_size * dim)`` working set fits
            within one worker's share of the memory budget
            (``autotune.per_worker_memory_budget_bytes(n_workers)``),
            AFTER first subtracting each worker's fixed resident
            footprint - the ``operator`` copy and setup arrays every
            worker holds for its whole lifetime, not just its current
            chunk (see ``_per_worker_resident_bytes``) - this second
            bound is necessary because up to ``n_workers`` chunks are
            live simultaneously here, unlike the single-process path,
            where only one chunk's working set is ever live at a time.
            Pass an explicit value to override either bound.
        n_workers: If ``None`` (default), uses
            ``_detect_available_worker_count()`` -
            ``len(os.sched_getaffinity(0))`` where available (Linux),
            not ``os.cpu_count()``, so a cgroup/cpuset-restricted HPC
            job is not over-subscribed. Pass an explicit value to
            override (e.g. to leave headroom for other work on a
            shared node).
        atol: Same as ``fwht_pauli_terms``.
        assume_hermitian: Same as ``fwht_pauli_terms_iter`` - checked
            per-chunk, same all-or-nothing-per-chunk (not
            all-or-nothing-per-operator) contract; see that function's
            own docstring for the difference from ``fwht_pauli_terms``.
        checkpoint_path: If given, checkpoints are written as the same
            binary chunk-framed format ``fwht_pauli_terms``/
            ``fwht_pauli_terms_iter`` use - one shared writer
            (``_append_checkpoint_frame``) and one shared reader
            (``_iter_checkpoint_frames``) on both paths, so a
            checkpoint written by either is resumable by the other
            (see ``docs/tutorial.md``). Both paths use the same
            append-only progress marker: one 8-byte little-endian
            ``u64`` record per completed chunk, appended to a sibling
            ``<checkpoint_path>.progress.json``/
            ``.parallel_progress.json`` file
            (``_append_progress_record``), and recovered by reading
            whole records into a set and discarding any torn trailing
            record (``_read_completed_indices``). What differs is only
            how each path turns that set into a resume point: the
            sequential path derives its single ``next_chunk`` index as
            ``max(recovered) + 1``, valid because chunks finish
            strictly in order, while this function uses the recovered
            set directly, because parallel workers finish out of
            order. The two progress markers use distinct file suffixes
            so they never collide; resume here re-submits every chunk
            index not already in that set, regardless of position.

    Yields:
        One ``dict`` per completed chunk, same value-type contract as
        ``fwht_pauli_terms_iter``. **Order is not guaranteed to match
        chunk order** - chunks are yielded as workers complete them,
        which depends on runtime scheduling, not input position. A
        caller that needs chunk-order output should sort/buffer
        itself; most callers (writing to disk, accumulating into an
        unordered structure, filtering) do not care about order.

    Raises:
        ValueError: Same conditions as ``fwht_pauli_terms_iter``,
            raised immediately for the shape/power-of-two check (before
            any worker starts); the ``assume_hermitian`` violation
            case is instead raised (as a chunk-processing exception,
            re-raised in the main process) once the offending chunk's
            worker task completes - unlike the sequential generator,
            this does not guarantee every chunk submitted *before* the
            offending one has already been yielded to the caller by
            the time it raises, since chunks do not complete in
            submission order.
    """
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    # NOTE: `autotune` is NOT imported here - `_recommended_parallel_chunk_size`
    # (called below only when chunk_size is None) already imports it
    # lazily itself. An unconditional import here was pure dead weight:
    # autotune's own first-import cost (measured ~9-20M instructions,
    # dominated by the compiled cache_probe extension) was paid on
    # EVERY call to this function regardless of whether chunk_size was
    # ever actually auto-computed - the same class of bug the
    # scipy.sparse lazy-import fix addressed earlier in this
    # investigation, found by the same instructions:u-driven work
    # analysis.

    (
        operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
    ) = _prepare_operator_for_fwht(
        operator
    )
    active_x, inverse = _active_x_and_inverse(x_nz, dim)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]

    # See _resolve_n_workers's own docstring for the measurement
    # behind the physical-core-count default this applies when
    # n_workers is None.
    n_workers = _resolve_n_workers(n_workers)

    if chunk_size is None:
        fixed_resident_bytes = _per_worker_resident_bytes(
            operator, is_sparse_input, len(p_nz)
        )
        chunk_size = _recommended_parallel_chunk_size(
            dim, n_workers, fixed_resident_bytes
        )

    n_workers = max(1, min(n_workers, max(1, (n_active + chunk_size - 1) // chunk_size)))

    # Same fully-dense fast path as parallel_decompose_arrays (see
    # that function's own comment for the full rationale and the
    # measurement behind it: this sort was ~40-57% of total time on
    # dense random Hermitian input before being skipped there).
    # _parallel_worker_chunk's fast path reads `operator` directly per
    # chunk and never touches sorted_inverse/sorted_p_nz/sorted_q_nz/
    # sorted_values in that case, so building them here - and then
    # pickling them into every worker process via ProcessPoolExecutor's
    # initargs - is pure waste for dense input, on top of the sort
    # cost itself.
    is_fully_dense = (not is_sparse_input) and n_active == dim
    if is_fully_dense:
        sorted_inverse = np.empty(0, dtype=inverse.dtype)
        sorted_p_nz = np.empty(0, dtype=p_nz.dtype)
        sorted_q_nz = np.empty(0, dtype=q_nz.dtype)
        sorted_values = np.empty(0, dtype=values_nz.dtype)
    else:
        # kind="quicksort", not "stable" - see parallel_decompose_arrays's
        # own comment: no downstream use needs original-order
        # preservation within a bucket of equal x-values.
        order = np.argsort(inverse, kind="quicksort")
        sorted_inverse = inverse[order]
        sorted_p_nz = p_nz[order]
        sorted_q_nz = q_nz[order]
        sorted_values = values_nz[order]

    chunk_starts = list(range(0, n_active, chunk_size))
    completed_indices, checkpoint_frames = _load_parallel_checkpoint(checkpoint_path)
    idx_dtype = _index_dtype_for_dim(dim)
    chunk_sink = _make_chunk_sink(checkpoint_path)
    if checkpoint_frames is not None:
        for ck_x, ck_z, ck_coeff in checkpoint_frames:
            labels = _pauli_label_batch(ck_x, ck_z, n_qubits)
            if assume_hermitian:
                yield _build_real_terms(labels, ck_coeff, atol)
            else:
                yield {
                    label: complex(c)
                    for label, c in zip(labels, ck_coeff.tolist())
                }

    pending = [
        (i, start, min(start + chunk_size, n_active))
        for i, start in enumerate(chunk_starts)
        if i not in completed_indices
    ]
    if not pending:
        return

    # Bounded submission - a REAL bug found by direct measurement
    # (2026-09-02): submitting every chunk as a task up front
    # (pool.submit for all
    # of `pending`, often thousands of tasks at real N) lets completed
    # workers' results pile up in the pool's IPC/result queue faster
    # than this single-threaded as_completed loop drains them - the
    # backlog of already-computed-but-not-yet-consumed (x, z, coeff)
    # arrays is NOT bounded by chunk_size or per_worker_memory_budget_
    # bytes at all, and grows with n_workers (more workers finish
    # chunks faster, the drain rate here does not increase to match) -
    # measured real RSS scaling from ~5 GiB (n_workers=1) to ~25 GiB
    # (n_workers=8) at N=150, confirming this, not the chunk_size
    # working set, was the dominant memory cost. Keeping at most
    # roughly one in-flight task per worker (plus a small pipelining
    # margin) bounds the backlog to O(n_workers), matching the
    # O(chunk_size * dim) per-task footprint the memory-budget
    # division above was already designed to control.
    max_in_flight = max(1, 2 * n_workers)

    # CPU-pinning fix, found necessary by direct measurement: without
    # this, ProcessPoolExecutor workers are freely migrated by the
    # Linux scheduler across ALL logical CPUs,
    # confirmed via direct ps -o psr sampling to cause hyperthread-
    # sibling collisions (two workers on the same physical core at
    # once) at every n_workers value tested, not just when n_workers
    # exceeds the physical core count. pin_cpus is one representative
    # logical CPU per physical core (None if unavailable - non-Linux,
    # or the physical-core probe itself failed); next_pin_index is a
    # cross-process shared counter each worker atomically increments
    # on startup to claim a distinct entry (ProcessPoolExecutor's
    # initializer gives every worker identical initargs, with no
    # built-in per-worker ordinal of its own).
    pin_cpus = _physical_core_representative_cpus()
    next_pin_index = multiprocessing.Value("i", 0)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(
            operator, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
            sorted_values,
            active_x, dim, n_qubits, z_indices, atol, pin_cpus, next_pin_index,
        ),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_next() -> bool:
            item = next(pending_iter, None)
            if item is None:
                return False
            chunk_index, chunk_start, chunk_end = item
            in_flight.add(pool.submit(_parallel_worker_chunk, chunk_index, chunk_start, chunk_end))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()
                _submit_next()  # keep in_flight near max_in_flight as work drains

                chunk_sink.write_parallel_chunk(
                    completed_indices, chunk_index,
                    chunk_x_out, z_idx, chunk_coeff_out, idx_dtype,
                )

                labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)
                if assume_hermitian:
                    yield _build_real_terms(labels, chunk_coeff_out, atol)
                else:
                    yield {
                        label: complex(c) for label, c in zip(labels, chunk_coeff_out.tolist())
                    }


def parallel_decompose_arrays(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    chunk_size: int | None = None,
    n_workers: int | None = None,
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    checkpoint_path: str | Path | None = None,
    executor: str = "auto",
    assume_dense: bool = False,
    eager_threads: bool = True,
    chunk_plan: dict | None = None,
) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
    """Multi-core decomposition yielding raw ``(x, z, coeff)`` arrays.

    Identical machinery to ``parallel_decompose`` (same chunking, same
    auto-tuning, same bounded submission, same CPU pinning, same
    checkpoint format) with one difference: it yields each chunk's raw
    arrays instead of building a ``dict[str, complex]`` from them.

    That difference is the whole point. Building ~91.6M Python ``str``
    objects and dict entries at N=150 is ~82% of ``parallel_decompose``'s
    total runtime, all of it in the single parent process, which caps
    its speedup at ~1.21x no matter how many cores are available
    (measured: best-ever 1.284x, and 8 workers gives 1.100x). Removing
    that work from the drain loop was measured to restore real
    multi-core scaling - 2.191x, statistically indistinguishable from a
    control doing no drain-side work at all.

    Use ``terms_from_arrays`` to render any chunk (or a filtered subset
    of one) to the usual label -> coefficient dict.

    Yields:
        ``(x, z, coeff)`` per completed chunk: two integer arrays of
        symplectic bitmasks and one ``complex128`` coefficient array,
        all the same length. **Order is not guaranteed to match chunk
        order** - same contract as ``parallel_decompose``. A chunk with
        no surviving terms yields three empty arrays rather than being
        skipped, so chunk count is stable.

    Raises:
        ValueError: If ``assume_hermitian=True`` and any coefficient
            has a non-negligible imaginary part. Checked per chunk, so
            this can raise *after* earlier chunks have been yielded -
            the same partial-yield-then-error contract
            ``fwht_pauli_terms_iter`` documents. Coefficients are kept
            ``complex128`` across the process boundary precisely so
            this check remains possible.

    ``assume_dense`` (default ``False``): skips the ``np.nonzero``
    sparsity-DETECTION scan entirely when ``True`` (and ``operator``
    is a plain ndarray, never scipy.sparse) - a pure performance
    choice, not a correctness-risking promise. Correct for ANY actual
    sparsity pattern, including exact-zero entries or a mostly-zero
    operator (verified directly: a zeroed single cell, a zeroed
    row/column, and a 90%-zero random pattern all produced
    bit-identical output to auto-detection) - the fast path this
    enables reads ``operator[p, q]`` directly per chunk rather than
    scattering from a precomputed nonzero list, so it never depends on
    which cells happen to be zero. Skipping the detection scan is
    where the real cost lives: paulikit does not know a priori whether
    input is sparse or dense and must discover it on every call, while
    a tool built only for dense input (e.g. ``pauli_lcu``) never pays
    this at all. Measured directly (`perf stat` ``instructions:u``):
    that scan alone costs ~594-688M instructions at dim=1024 -
    comparable to `pauli_lcu`'s entire decomposition (689M
    instructions) at the same size.

    The only real downside of setting this on a genuinely SPARSE
    operator is performance, not correctness: the fast path always
    reads and transforms every cell (``dim*dim`` work), so it is
    worthwhile only when ``operator`` is dense enough that skipping
    the detection scan is worth paying for full-array reads and
    transforms instead of the sparse-aware scatter path. Leave this at
    the default whenever sparsity is unknown or expected.

    ``eager_threads`` (default ``True``, only affects ``executor ==
    "thread"``): forces all ``n_workers`` OS threads to exist before
    the first real chunk is submitted, instead of
    ``ThreadPoolExecutor``'s own default lazy behavior (a new OS
    thread is spawned only when ``submit()`` finds no idle thread
    already waiting - confirmed directly against the CPython 3.12
    source, ``_adjust_thread_count``). Measured directly (entry/exit
    instrumentation on a real N=150, n_workers=5 run - see
    paulikit-manuscript/debug/SESSION_QA.md Phase 10): without this,
    the first 5 chunks start staggered by ~350-450us each rather than
    together, because the priming loop's first 5 ``pool.submit()``
    calls each force a fresh ``pthread_create`` - a real, one-time
    ~1.5-2ms cost, small relative to N=150's own ~1.5-3s total
    runtime (~0.05-0.1%) but real and now precisely measured. Setting
    this to ``True`` submits ``n_workers`` barrier-synchronized primer
    tasks immediately after the pool is constructed (each blocks on a
    shared ``threading.Barrier(n_workers)`` until every worker thread
    has actually started, so none can go idle and get reused by a
    later ``submit()`` before the pool is fully warmed - verified this
    detail matters: a naive "submit n_workers trivial no-ops" attempt
    without a barrier only spawned 2 of 5 threads in one measured
    trial, since fast tasks let a thread finish and go idle before the
    next submit() even runs). No effect on ``executor == "process"``
    (this parameter is not read there) and no effect on the single-
    worker fast path (``n_workers == 1`` never constructs a
    ``ThreadPoolExecutor`` at all, so there is nothing to prime).
    Output is bit-identical either way - this only changes WHEN
    threads are created, never what they compute. Defaults to
    ``True`` since the staggered-startup cost it removes is real and
    free to remove (paulikit-manuscript/debug/SESSION_QA.md Phase 10);
    pass ``False`` to restore the old lazy spin-up if that ever
    matters (e.g. profiling the pool's own startup behavior).
    """
    # NOTE: `autotune` is NOT imported here - see parallel_decompose's
    # own identical note. `_recommended_parallel_chunk_size` (called
    # below only when chunk_size is None) already imports it lazily.
    #
    # NOTE: `multiprocessing`/`concurrent.futures` are NOT imported
    # here either, deliberately - they are only used inside the
    # executor branches below (`if executor == "thread":` /
    # the ProcessPoolExecutor branch after it), never on the
    # `n_workers == 1` fast path this function also has, which
    # `return`s before reaching either. Measured directly (perf stat
    # instructions:u): importing `multiprocessing` plus
    # `concurrent.futures`'s four names costs ~3.8e7 instructions on a
    # cold process - this was a real, previously uncounted cost paid
    # by EVERY n_workers=1 call (the common case for the dense
    # benchmark this whole investigation is about), for machinery that
    # call never uses. The exact same bug class as the scipy.sparse
    # and dead-autotune-import fixes earlier in this investigation -
    # found by re-measuring the n_workers=1-vs-direct-generator-call
    # gap with perf stat instead of the wall-clock timers that
    # self-contradicted when this was first investigated.

    if executor not in ("auto", "process", "thread"):
        raise ValueError(
            f"executor must be 'auto', 'process' or 'thread', "
            f"got {executor!r}"
        )
    if executor == "auto":
        # Threads are the better drain ONLY when the compiled kernels
        # are present, because that is what releases the GIL. The
        # difference is not marginal in either direction. Measured at
        # N=100, 4 workers:
        #
        #   kernels present : thread 0.313s vs process 2.009s (6.4x)
        #   kernels absent  : thread 2.460s vs process 1.708s (0.69x)
        #
        # Without them the NumPy fallback holds the GIL for the bulk
        # of each chunk, so threads serialise and additionally pay
        # contention; processes remain the right choice. Choosing per
        # build rather than globally is what makes this safe to
        # default.
        executor = (
            "thread"
            if (_wht_native is not None and _coeffs_native is not None)
            else "process"
        )

    (
        operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz, values_nz
    ) = _prepare_operator_for_fwht(
        operator, assume_dense=assume_dense
    )
    active_x, inverse = _active_x_and_inverse(
        x_nz, dim, is_fully_dense=assume_dense and not is_sparse_input
    )
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]

    # See _resolve_n_workers's own docstring for the measurement
    # behind the physical-core-count default this applies when
    # n_workers is None.
    n_workers = _resolve_n_workers(n_workers)

    if chunk_size is None:
        fixed_resident_bytes = _per_worker_resident_bytes(
            operator, is_sparse_input, len(p_nz)
        )
        chunk_size = _recommended_parallel_chunk_size(
            dim, n_workers, fixed_resident_bytes
        )

    n_workers = max(1, min(n_workers, max(1, (n_active + chunk_size - 1) // chunk_size)))

    # SINGLE-WORKER FAST PATH: skip the executor entirely.
    #
    # With n_workers == 1 there is nothing to parallelize -
    # ThreadPoolExecutor still pays for a worker thread, a Future per
    # chunk, wait(FIRST_COMPLETED), and the submit/result bookkeeping,
    # none of which buys anything when only one chunk is ever
    # in flight. Falsified directly before building this (per instinct
    # from the user: "could not we get rid of that by creating a
    # dedicated singlecore path... test that first, falsification
    # first, right?") - a synchronous loop over _parallel_worker_chunk
    # with no executor at all measured 22.4% fewer instructions than
    # the threaded path at n_workers=1 on dense random Hermitian input
    # (perf stat instructions:u, dim=1024: 2.104e8 -> 1.633e8).
    #
    # Rather than build a THIRD per-chunk body, this delegates to
    # `_iter_chunked_coefficients` - the sequential streaming API's own
    # generator (already fixed with the identical is_fully_dense fast
    # path, see fwht_pauli_coefficients), which turned out to be
    # LEANER than the throwaway synchronous-loop test above (1.539e8 -
    # no _parallel_worker_state global save/restore dance, no
    # ProcessPoolExecutor-shaped state dict to build for a case that
    # will never use it). Same (x, z, coeff) yield contract as this
    # function's own, so no chunk-level transformation is needed - only
    # the checkpoint and Hermiticity-check wrapping this function's own
    # contract already does around every other branch.
    #
    # Gated on `checkpoint_path is None` too: `_iter_chunked_coefficients`
    # uses `_load_checkpoint`/`_append_checkpoint_chunk` (the
    # SEQUENTIAL checkpoint format - a distinct progress-file suffix,
    # see `_checkpoint_progress_path`'s own docstring for why the two
    # formats never collide), not this function's own
    # `_load_parallel_checkpoint`/`_append_parallel_checkpoint_chunk`.
    # A caller who checkpoints across multiple calls at different
    # `n_workers` values (e.g. resuming at n_workers=1 a run that
    # crashed at n_workers=4) would silently fail to find their
    # existing checkpoint if this fast path used it unconditionally -
    # not data corruption, but surprising, avoidable redundant
    # recomputation. Falling through to the executor path (with its
    # own, already-correct checkpoint handling) whenever checkpointing
    # is requested at all sidesteps the question entirely, at the cost
    # of this fast path only applying to the common uncheckpointed case.
    if n_workers == 1 and checkpoint_path is None:
        for chunk_x, chunk_z, chunk_coeff in _iter_chunked_coefficients(
            operator, is_sparse_input, active_x, inverse, p_nz, q_nz, values_nz,
            dim, n_qubits, n_active, z_indices, chunk_size, atol, checkpoint_path,
            chunk_plan=chunk_plan,
        ):
            if assume_hermitian:
                _check_hermitian_violation(
                    chunk_coeff, atol, chunk_x, chunk_z, n_qubits
                )
            yield chunk_x, chunk_z, chunk_coeff
        return

    # Only imported here, past the n_workers==1 fast path's own
    # return above - see this function's opening comment for why
    # (measured ~3.8e7 instructions on a cold process, previously paid
    # unconditionally by every call including the one that never uses
    # any of these names).
    import multiprocessing
    from concurrent.futures import (
        FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait,
    )

    is_fully_dense = (not is_sparse_input) and n_active == dim
    if is_fully_dense:
        # FULLY DENSE input: _parallel_worker_chunk's fast path reads
        # `operator` directly (see its own docstring) and never touches
        # sorted_inverse/sorted_p_nz/sorted_q_nz/sorted_values - so
        # building them here at all is pure waste. This global sort
        # (previously unconditional) was measured to be ~40-57% of
        # total parallel_decompose_arrays time on dense random
        # Hermitian input even AFTER the worker-side fast path was
        # added, because this setup step still ran regardless of
        # whether any worker would ever read its output - skipping it
        # entirely, not just cheapening it, is what actually removes
        # that cost. Empty placeholders below are never read by the
        # fast path; they exist only so _parallel_worker_init's shared
        # state dict has a consistent shape for both branches.
        sorted_inverse = np.empty(0, dtype=inverse.dtype)
        sorted_p_nz = np.empty(0, dtype=p_nz.dtype)
        sorted_q_nz = np.empty(0, dtype=q_nz.dtype)
        sorted_values = np.empty(0, dtype=values_nz.dtype)
    else:
        # kind="quicksort" (NumPy's default), not "stable": every
        # downstream use of sorted_inverse is a np.searchsorted range
        # query (see _parallel_worker_chunk), which only needs sorted
        # VALUES, not original-order preservation within a bucket of
        # equal x-values - each (p, q, value) triple travels through
        # `order` together and is later scattered into
        # gathered_chunk[row, q] by q, so the physical (p, q) -> value
        # correspondence is exact regardless of which same-valued entry
        # lands first. Verified directly: reconstructing {(p, q): value}
        # from the quicksort-ordered arrays is identical to the
        # stable-sort ordering's reconstruction.
        order = np.argsort(inverse, kind="quicksort")
        sorted_inverse = inverse[order]
        sorted_p_nz = p_nz[order]
        sorted_q_nz = q_nz[order]
        sorted_values = values_nz[order]

    chunk_starts = list(range(0, n_active, chunk_size))
    if chunk_plan is not None:
        chunk_plan["n_chunks"] = len(chunk_starts)
    completed_indices, checkpoint_frames = _load_parallel_checkpoint(checkpoint_path)
    idx_dtype = _index_dtype_for_dim(dim)
    chunk_sink = _make_chunk_sink(checkpoint_path)
    if checkpoint_frames is not None:
        for ck_x, ck_z, ck_coeff in checkpoint_frames:
            if assume_hermitian:
                _check_hermitian_violation(
                    ck_coeff, atol, ck_x, ck_z, n_qubits
                )
            yield ck_x, ck_z, ck_coeff

    pending = [
        (i, start, min(start + chunk_size, n_active))
        for i, start in enumerate(chunk_starts)
        if i not in completed_indices
    ]
    if not pending:
        return

    # Bounded submission - a REAL bug found by direct measurement
    # (2026-09-02): submitting every chunk as a task up front
    # (pool.submit for all
    # of `pending`, often thousands of tasks at real N) lets completed
    # workers' results pile up in the pool's IPC/result queue faster
    # than this single-threaded as_completed loop drains them - the
    # backlog of already-computed-but-not-yet-consumed (x, z, coeff)
    # arrays is NOT bounded by chunk_size or per_worker_memory_budget_
    # bytes at all, and grows with n_workers (more workers finish
    # chunks faster, the drain rate here does not increase to match) -
    # measured real RSS scaling from ~5 GiB (n_workers=1) to ~25 GiB
    # (n_workers=8) at N=150, confirming this, not the chunk_size
    # working set, was the dominant memory cost. Keeping at most
    # roughly one in-flight task per worker (plus a small pipelining
    # margin) bounds the backlog to O(n_workers), matching the
    # O(chunk_size * dim) per-task footprint the memory-budget
    # division above was already designed to control.
    max_in_flight = max(1, 2 * n_workers)

    # CPU-pinning fix, found necessary by direct measurement: without
    # this, ProcessPoolExecutor workers are freely migrated by the
    # Linux scheduler across ALL logical CPUs,
    # confirmed via direct ps -o psr sampling to cause hyperthread-
    # sibling collisions (two workers on the same physical core at
    # once) at every n_workers value tested, not just when n_workers
    # exceeds the physical core count. pin_cpus is one representative
    # logical CPU per physical core (None if unavailable - non-Linux,
    # or the physical-core probe itself failed); next_pin_index is a
    # cross-process shared counter each worker atomically increments
    # on startup to claim a distinct entry (ProcessPoolExecutor's
    # initializer gives every worker identical initargs, with no
    # built-in per-worker ordinal of its own).
    if executor == "thread":
        # THREADED DRAIN - static per-worker partitioning ("Design A",
        # paulikit-manuscript/debug/SESSION_QA.md Phase 15/15.3/23b).
        #
        # Threads work here for one specific reason: both compiled
        # kernels release the GIL, so `_parallel_worker_chunk`'s real
        # cost runs genuinely concurrently. Measured per chunk, the
        # GIL-held part (the zeroed block, the value slice, the
        # scatter) is 13.0us against 374.6us inside the kernels - a
        # serial fraction of f = 0.0336.
        #
        # Measured scaling (perf cycles, which unlike wall clock on
        # this machine are frequency-invariant): 1.87x on 2 threads
        # and 3.44x on 4, against Amdahl ceilings of 1.93x and 3.63x
        # for that f. Threading costs only 7% and 16% extra cycles
        # respectively.
        #
        # Why it is worth having: with the transform compiled, a chunk
        # costs ~0.5ms while a ProcessPoolExecutor round trip costs
        # ~1.33ms, so the process path became a net loss at these
        # sizes (0.94x at N=100, 0.81x at N=150, at 3-5x the CPU).
        # Threads pay no pickling at all - the arrays never leave this
        # address space.
        #
        # Static partitioning, not the dynamic bounded-submission
        # drain the process path below still uses: `pending` is split
        # into exactly `n_workers` contiguous slices up front, and
        # each worker gets exactly ONE `pool.submit()` call for its
        # whole slice, iterating it internally with a plain Python
        # `for` loop - zero further cross-thread `submit()` calls
        # after the initial `n_workers`. This removes
        # `ThreadPoolExecutor`'s shared, lock-contended work queue
        # from the hot path almost entirely (n_workers submissions
        # instead of len(pending) - 5 instead of 5595 at N=150,
        # n_workers=5), which is the actual mechanism the dynamic
        # path's contention lived in. Measured (perf stat
        # --no-inherit, 3 reps, N=150 n_workers=5): 8.6% fewer
        # instructions, 13.3% fewer cycles, IPC improved (not
        # degraded), and run-to-run variance roughly two orders of
        # magnitude tighter than the dynamic path (Phase 15.3) - a
        # real, unambiguous, not marginal win on every axis measured,
        # fully committed to here with no dynamic fallback: every
        # chunk-size/worker-count configuration measured across this
        # investigation (N=150, N=200, chunk_size sweeps) found
        # per-chunk cost uniform enough that static partitioning wins
        # outright and the case for keeping dynamic load-balancing
        # never materialized.
        #
        # Falsified before being ported to the ProcessPoolExecutor
        # path below: this design's streaming-results mechanism
        # (`queue.SimpleQueue`) only works within one process's shared
        # memory, and the process path's actually-dominant cost is
        # per-chunk pickling of `_CallItem`/`_ResultItem` (paid once
        # per chunk regardless of submit() call count) - Design A's
        # saving (fewer submissions) does not touch that cost, so it
        # is not applicable there without a wholly different,
        # unmeasured design. Scoped to the thread path only,
        # unconditionally, matching `eager_threads`'s own scope.
        #
        # Everything else is deliberately unchanged: same chunking,
        # same atol, same checkpoint format, same streamed/memory-
        # bounded yield contract (results still flow out one chunk at
        # a time, not collected then returned).
        import queue as _stream_queue
        from collections import deque as _stream_deque

        global _parallel_worker_state
        saved_state = _parallel_worker_state
        # Threads share the parent's memory, so the per-worker state
        # the process path ships through `initargs` is simply set here
        # and read by the same `_parallel_worker_chunk`.
        #
        # Pinning, unlike an earlier version of this comment claimed:
        # `os.sched_setaffinity(0, {cpu})` operates on the CALLING
        # THREAD, not the whole process, when called from inside one -
        # confirmed directly (three threads each pinned themselves to
        # a distinct CPU, and `sched_getaffinity(0)` read back from
        # each showed its own single-CPU mask while the main thread's
        # own affinity stayed the full unrestricted set throughout).
        # Assigning each of the n_workers threads a DIFFERENT
        # `pin_cpus` entry (one representative logical CPU per
        # physical core, same list `_physical_core_representative_
        # cpus` already gives the process path) spreads threads across
        # distinct cores - the opposite of serialising them onto one.
        # Measured directly (paulikit-manuscript/debug/SESSION_QA.md
        # Phase 29, 5 reps, Welch's t-test): at n_workers ==
        # physical-core-count, pinning makes no measurable difference
        # (the scheduler already places threads well when nothing
        # forces a collision); at n_workers > physical-core-count
        # (forced hyperthread-sibling oversubscription, the real
        # N=150/n_workers=5 config on a 4-core machine), pinning gives
        # a real, statistically significant ~5.2% win (p=0.0008) with
        # roughly 3x tighter run-to-run variance - the same "tighter
        # and faster" signature Design A's own submission-queue fix
        # showed. Assigned by SLICE index (fixed and known up front,
        # since Design A already assigns one contiguous slice per
        # worker before any thread starts), not a shared/racing
        # counter like the process path's first-come-first-served
        # `next_pin_index` - there is nothing to race here.
        pin_cpus = _physical_core_representative_cpus()

        _parallel_worker_init(
            operator, is_sparse_input, sorted_inverse, sorted_p_nz,
            sorted_q_nz, sorted_values, active_x, dim, n_qubits,
            z_indices, atol, None, None,
        )
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                if eager_threads:
                    # Force all n_workers OS threads to exist now,
                    # before any real chunk is submitted - see this
                    # function's own docstring for why (measured
                    # ~350-450us per-thread stagger otherwise) and why
                    # a barrier specifically (a naive "submit
                    # n_workers no-ops" without one under-spawns,
                    # verified directly). A Barrier(n) call that never
                    # collects n parties blocks forever, so an explicit
                    # timeout turns any future mismatch between the
                    # pool's real worker count and n_workers into a
                    # loud BrokenBarrierError instead of a silent hang
                    # - defensive, since n_workers is validated earlier
                    # in this function (_resolve_n_workers,
                    # max_in_flight's own max(1, ...) floor) and should
                    # never actually mismatch here.
                    _eager_barrier = threading.Barrier(n_workers)

                    def _eager_prime() -> None:
                        _eager_barrier.wait(timeout=30.0)

                    _eager_futures = [
                        pool.submit(_eager_prime) for _ in range(n_workers)
                    ]
                    for _f in _eager_futures:
                        _f.result(timeout=30.0)

                # Split `pending` into n_slices contiguous slices,
                # CAPACITY-WEIGHTED by physical core, not equal-sized.
                #
                # Why: when n_workers does not evenly divide the
                # physical core count, `pin_cpus[slice_index %
                # len(pin_cpus)]` (below) necessarily pins more than
                # one worker thread to the same core for at least one
                # core - e.g. n_workers=5 on 4 cores pins slices 0 and
                # 4 to the SAME core while slices 1-3 each get their
                # own. Measured directly (paulikit-manuscript/debug/
                # SESSION_QA.md Phase 39/40): giving every slice the
                # SAME chunk count in that situation makes the two
                # shared-core slices take ~1.57x as long, in every
                # rep, not a chunk-content effect (a single-threaded,
                # uncontended sweep of the identical chunk ranges
                # showed only a 1.03x/noise-level difference) - a
                # direct, mechanical consequence of two full workloads
                # timesharing one core while neighboring cores get a
                # dedicated one each.
                #
                # Fix: size each worker's slice inversely to how many
                # workers share its assigned core (a worker alone on
                # its core gets a full share; two workers sharing a
                # core each get half), so the TOTAL chunk count
                # assigned to any one physical core stays balanced
                # even though individual worker THREAD counts per core
                # are not. This is the naive 1/k model (k = workers
                # sharing that core), not calibrated to the measured
                # ~1.57x contention factor - deliberately simple and
                # auditable; refine only if real measurement after
                # this change shows the naive split under/over-
                # corrects (see Phase 40's own note that 1/k assumes
                # perfectly serialized sharing, while GIL-released
                # kernels partially interleave, so the real optimum
                # may sit between equal-count and a full 1/k split).
                #
                # n_slices is capped at len(pending) so a tiny job
                # (fewer pending chunks than n_workers) never creates
                # empty slices - unchanged from before.
                n_slices = min(n_workers, len(pending))
                if pin_cpus:
                    _workers_per_core: dict[int, int] = {}
                    for _w in range(n_slices):
                        _c = pin_cpus[_w % len(pin_cpus)]
                        _workers_per_core[_c] = _workers_per_core.get(_c, 0) + 1
                    _capacity = [
                        1.0 / _workers_per_core[pin_cpus[_w % len(pin_cpus)]]
                        for _w in range(n_slices)
                    ]
                else:
                    # No pinning available (non-Linux, or the physical-
                    # core probe failed) - every slice gets equal
                    # capacity, same as the original unweighted split.
                    _capacity = [1.0] * n_slices
                _total_capacity = sum(_capacity)
                slice_sizes = [
                    round(len(pending) * c / _total_capacity) for c in _capacity
                ]
                # Rounding can drift the total off len(pending) by a
                # few items - correct on the LAST slice, matching the
                # original code's own "only the last slice may be
                # shorter" contract.
                slice_sizes[-1] += len(pending) - sum(slice_sizes)

                # Each worker's slice is a deque, not a plain list, so
                # an idle worker can STEAL remaining work from another
                # worker's deque once its own runs dry - see the work-
                # stealing rationale below. `collections.deque` is
                # documented (and empirically verified, paulikit-
                # manuscript/debug/SESSION_QA.md Phase 42) thread-safe
                # for concurrent `popleft()`(front)/`pop()`(back) on
                # the SAME deque with no explicit lock: each op is a
                # single GIL-atomic step, and the deque's own
                # implementation keeps front/back operations from
                # corrupting each other. Only the OWNING worker ever
                # calls `popleft()` on its own deque; only OTHER
                # workers (thieves) call `pop()` on it.
                slices: list[_stream_deque] = []
                _pos = 0
                for _size in slice_sizes:
                    slices.append(_stream_deque(pending[_pos:_pos + _size]))
                    _pos += _size

                # Thread-safe, unbounded, no lock-tuning needed -
                # results stream back one chunk at a time as each
                # worker finishes it, with a sentinel per finished
                # slice so this consumer loop knows when all
                # n_slices workers are done without polling.
                results_q: _stream_queue.SimpleQueue = _stream_queue.SimpleQueue()
                _SLICE_DONE = object()

                def _steal(my_index: int):
                    """Take one item from whichever OTHER worker's
                    deque currently holds the most work, trying the
                    next-fullest candidate if a race empties the
                    chosen one first (no lock: a lost race is just an
                    `IndexError`, not a correctness problem - each
                    individual `pop()`/`popleft()` is already atomic).
                    Fullest-victim, not Cilk's classic random-victim:
                    measured directly (SESSION_QA.md Phase 42, Monte
                    Carlo, 100k trials/config) that at this workload's
                    real worker-count range (4-16; oversubscription
                    beyond physical-core-count already measured worse
                    elsewhere in this investigation), fullest-victim
                    finds real work in exactly one lookup with 100%
                    success whenever any exists, while random-victim's
                    success rate degrades sharply (as low as ~79% in
                    one tested config) exactly when most workers are
                    already drained - the tail-of-run scenario this
                    mechanism exists to fix. The O(n_slices) length
                    scan this requires is real but negligible at this
                    scale (measured ~1-2us at n_slices=4-16, ~0.0002%
                    of one chunk's own ~700us cost) - Cilk's own
                    justification for random selection (avoiding that
                    scan) targets a scale of dozens-to-thousands of
                    workers this single-machine, physical-core-bound
                    design never reaches.
                    Returns the stolen `(chunk_index, chunk_start,
                    chunk_end)` tuple, or `None` if every other deque
                    is confirmed empty.
                    """
                    victims = sorted(
                        (j for j in range(len(slices)) if j != my_index),
                        key=lambda j: -len(slices[j]),
                    )
                    for victim in victims:
                        try:
                            return slices[victim].pop()
                        except IndexError:
                            continue
                    return None

                def _run_slice(slice_index: int, my_slice) -> None:
                    if pin_cpus:
                        _pin_current_process_to_cpu(
                            pin_cpus[slice_index % len(pin_cpus)]
                        )
                    while True:
                        try:
                            ci, cs_, ce_ = my_slice.popleft()
                        except IndexError:
                            item = _steal(slice_index)
                            if item is None:
                                break
                            ci, cs_, ce_ = item
                        results_q.put(_parallel_worker_chunk(ci, cs_, ce_))
                    results_q.put(_SLICE_DONE)

                slice_futures = [
                    pool.submit(_run_slice, i, s)
                    for i, s in enumerate(slices)
                ]

                done_slices = 0
                while done_slices < len(slices):
                    item = results_q.get()
                    if item is _SLICE_DONE:
                        done_slices += 1
                        continue
                    chunk_index, chunk_x_out, z_idx, chunk_coeff_out = item

                    chunk_sink.write_parallel_chunk(
                        completed_indices,
                        chunk_index, chunk_x_out, z_idx,
                        chunk_coeff_out, idx_dtype,
                    )

                    if assume_hermitian:
                        _check_hermitian_violation(
                            chunk_coeff_out, atol, chunk_x_out,
                            z_idx, n_qubits
                        )
                    yield chunk_x_out, z_idx, chunk_coeff_out

                # Propagate any exception raised inside a worker's
                # slice (e.g. a genuine non-Hermitian input the
                # per-chunk check above did not already catch and
                # raise first, or an unexpected kernel error) - same
                # contract as the dynamic path's future.result() call,
                # which would have raised at the point of iteration.
                for _f in slice_futures:
                    _f.result()
        finally:
            # Restore rather than clear: a worker PROCESS legitimately
            # holds state here, and this function can be called from
            # inside one.
            _parallel_worker_state = saved_state
        return

    pin_cpus = _physical_core_representative_cpus()
    next_pin_index = multiprocessing.Value("i", 0)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(
            operator, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
            sorted_values,
            active_x, dim, n_qubits, z_indices, atol, pin_cpus, next_pin_index,
        ),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_next() -> bool:
            item = next(pending_iter, None)
            if item is None:
                return False
            chunk_index, chunk_start, chunk_end = item
            in_flight.add(pool.submit(_parallel_worker_chunk, chunk_index, chunk_start, chunk_end))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()
                _submit_next()

                chunk_sink.write_parallel_chunk(
                    completed_indices, chunk_index,
                    chunk_x_out, z_idx, chunk_coeff_out, idx_dtype,
                )

                if assume_hermitian:
                    _check_hermitian_violation(
                        chunk_coeff_out, atol, chunk_x_out, z_idx, n_qubits
                    )
                yield chunk_x_out, z_idx, chunk_coeff_out


_WARNED_NO_NATIVE = False


def _pauli_label_batch(
    x_indices: NDArray[np.integer],
    z_indices: NDArray[np.integer],
    n_qubits: int,
    parallel: bool = False,
) -> list[str]:
    """Batch IXYZ labels for parallel arrays of (x, z) indices.

    Uses the compiled ``pauli_label_native`` extension when available
    (built from the same C kernel that was benchmarked against
    alternative bindings during development); falls back to the
    pure-Python ``pauli_label`` loop otherwise, since paulikit must
    stay pip-installable without a C++ toolchain. The fallback is
    NOT silent: paulikit's whole purpose is
    fast Pauli decomposition, so running the slow path unknowingly
    would defeat the point of the package - a warning fires once per
    process the first time the fallback is actually used.

    Args:
        parallel: If ``True`` and ``pauli_label_parallel_native`` is
            available (oneTBB), uses that module's
            ``pauli_label_batch_parallel`` instead of the serial C
            kernel. The C fill alone is ~3-4x faster under TBB at
            ~1.26M terms, but end-to-end ``list[str]`` construction
            still spends ~94% of wall time in Python ``str`` decode,
            so e2e speedup is ~1.03x. Left opt-in. If the parallel
            module is missing, falls back to serial native (or pure
            Python) — same labels, no error.
    """
    x_masks = np.asarray(x_indices, dtype=np.uint32)
    z_masks = np.asarray(z_indices, dtype=np.uint32)

    if parallel and _native_parallel is not None:
        return _native_parallel.pauli_label_batch_parallel(
            x_masks, z_masks, n_qubits
        )

    if _native is not None:
        return _native.pauli_label_batch(x_masks, z_masks, n_qubits)

    global _WARNED_NO_NATIVE
    if not _WARNED_NO_NATIVE:
        warnings.warn(
            "paulikit's compiled pauli_label fast path is not available "
            "(built with -Dnative=disabled, or a C compiler/Cython were "
            "missing at build time) - using the pure-Python pauli_label "
            "loop, which is substantially slower for large term counts. "
            "Rebuild paulikit with a C compiler and Cython >= 3.0 to get "
            "the compiled serial fast path (oneTBB is only needed for "
            "the optional parallel_labels fill).",
            stacklevel=3,
        )
        _WARNED_NO_NATIVE = True

    return [
        pauli_label(int(x), int(z), n_qubits)
        for x, z in zip(x_indices.tolist(), z_indices.tolist())
    ]
