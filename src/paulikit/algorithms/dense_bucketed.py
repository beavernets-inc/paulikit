# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Bucketed / spill / file-backed dense OperatorSource (opt-in).

Import this module only when constructing a bucketed dense source —
not from ``fwht``'s default import path.

Spill geometry (``spill_bucket_rows``) may be taller than the drain
tile (``chunk_size``). Spilled Pass-1 requires power-of-two
``spill_bucket_rows`` (and therefore ``chunk_size``) and bit-routes
``q = p ⊕ (x_lo + r)`` via ``pass1_scatter_native`` when built.
File-backed construction memmaps ``H`` so each cell is read once;
``gather_chunk`` reads thin drain slices so drain peak tracks
``chunk_size``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np


_pass1_native_mod = False  # False = unset; then module or None


def _pass1_native():
    """Lazy optional import — avoid loading the extension until spill."""
    global _pass1_native_mod
    if _pass1_native_mod is False:
        try:
            from paulikit._native import pass1_scatter_native as mod
            _pass1_native_mod = mod
        except ImportError:
            _pass1_native_mod = None
    return _pass1_native_mod


def _bucket_height(dim: int, bucket_rows: int, bucket_id: int) -> int:
    return min(bucket_rows, dim - bucket_id * bucket_rows)


def _bucket_path(spill_dir: Path, bucket_id: int) -> Path:
    return spill_dir / f"bucket_{bucket_id:06d}.c128"


def _write_bucket(path: Path, bucket) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(bucket, dtype=np.complex128)
    native = _pass1_native()
    if native is not None:
        native.write_c128_file(path, arr)
    else:
        arr.tofile(path)


def _read_bucket(path: Path, n_rows: int, dim: int):
    expected = n_rows * dim
    flat = np.fromfile(path, dtype=np.complex128, count=expected)
    if flat.size != expected:
        raise ValueError(
            f"bucket file {path} has {flat.size} complex128 values, "
            f"expected {expected} ({n_rows}×{dim})"
        )
    return np.ascontiguousarray(flat.reshape(n_rows, dim))


def _read_bucket_slice(path: Path, dim: int, row_offset: int, n_rows: int):
    """Read ``n_rows`` consecutive gather-rows from a spill tile file."""
    if row_offset < 0 or n_rows < 0:
        raise ValueError(
            f"row_offset={row_offset} and n_rows={n_rows} must be >= 0"
        )
    row_bytes = dim * np.dtype(np.complex128).itemsize
    nbytes = n_rows * row_bytes
    with Path(path).open("rb") as fh:
        fh.seek(row_offset * row_bytes)
        raw = fh.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(
            f"short read from {path}: got {len(raw)} bytes, "
            f"expected {nbytes} for rows [{row_offset}, {row_offset + n_rows})"
        )
    return np.frombuffer(raw, dtype=np.complex128).reshape(n_rows, dim).copy()


def _scatter_row_into_bucket(
    bucket,
    *,
    p: int,
    row,
    q_range,
    x_lo: int,
    x_hi: int,
) -> None:
    """Boolean-mask scatter (reference / non-power-of-two resident path)."""
    xs = p ^ q_range
    mask = (xs >= x_lo) & (xs < x_hi)
    if np.any(mask):
        bucket[xs[mask] - x_lo, q_range[mask]] = row[mask]


def _scatter_row_into_bucket_bit(
    bucket,
    *,
    p: int,
    row,
    x_lo: int,
    height: int,
) -> None:
    """Bit-indexed scatter for tiles with R = 2^k: q = p ⊕ (x_lo + r)."""
    r = np.arange(height, dtype=np.intp)
    q = p ^ (x_lo + r)
    bucket[r, q] = row[q]


def _is_power_of_two(n: int) -> bool:
    # Hacker's Delight: power-of-two iff n > 0 and (n & (n - 1)) == 0.
    return n > 0 and (n & (n - 1)) == 0


def _validate_spill_vs_chunk(
    chunk_size: int,
    spill_bucket_rows: int,
    *,
    bit_route: bool,
) -> None:
    if spill_bucket_rows < 1:
        raise ValueError(
            f"spill_bucket_rows must be >= 1, got {spill_bucket_rows}"
        )
    if spill_bucket_rows % chunk_size != 0:
        raise ValueError(
            f"spill_bucket_rows ({spill_bucket_rows}) must be a multiple "
            f"of chunk_size ({chunk_size}) so drain tiles do not cross "
            f"spill-file boundaries"
        )
    if not bit_route:
        return
    if not _is_power_of_two(spill_bucket_rows):
        raise ValueError(
            f"spill_bucket_rows must be a power of two (Pass-1 bit "
            f"routing), got {spill_bucket_rows}"
        )
    if not _is_power_of_two(chunk_size):
        raise ValueError(
            f"chunk_size must be a power of two when spilling "
            f"(required because spill_bucket_rows is a power of two and "
            f"must be a multiple of chunk_size), got {chunk_size}"
        )


class DenseBucketedSource:
    """Dense gather via scatter into fixed-size x-buckets.

    Without ``spill_dir``, all buckets stay in RAM (height =
    ``chunk_size``). With ``spill_dir``, Pass-1 writes one spill tile
    at a time (height ``spill_bucket_rows``, default ``chunk_size``)
    and ``gather_chunk`` reloads only the requested drain slice.
    Spilled geometry requires power-of-two ``spill_bucket_rows`` and
    ``chunk_size``.

    File-backed construction: :meth:`from_dense_file` (layouts A/B) or
    :meth:`from_complex128_file`. See ``docs/dense_out_of_core.md``.
    """

    __slots__ = (
        "_bucket_files",
        "_buckets_ram",
        "_cache",
        "_cache_lock",
        "_chunk_size",
        "_dim",
        "_max_resident_buckets",
        "_n_buckets",
        "_spill_bucket_rows",
        "_spill_dir",
    )

    def __init__(
        self,
        *,
        dim: int,
        chunk_size: int,
        n_buckets: int,
        buckets_ram,
        bucket_files,
        spill_dir,
        max_resident_buckets: int,
        spill_bucket_rows: int | None = None,
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if max_resident_buckets < 1:
            raise ValueError(
                f"max_resident_buckets must be >= 1, got {max_resident_buckets}"
            )
        spill_rows = (
            int(chunk_size)
            if spill_bucket_rows is None
            else int(spill_bucket_rows)
        )
        _validate_spill_vs_chunk(
            chunk_size,
            spill_rows,
            bit_route=(spill_dir is not None),
        )
        self._dim = int(dim)
        self._chunk_size = int(chunk_size)
        self._spill_bucket_rows = spill_rows
        self._n_buckets = int(n_buckets)
        self._buckets_ram = (
            list(buckets_ram)
            if buckets_ram is not None
            else [None] * self._n_buckets
        )
        self._bucket_files = (
            list(bucket_files)
            if bucket_files is not None
            else [None] * self._n_buckets
        )
        self._spill_dir = Path(spill_dir) if spill_dir is not None else None
        self._max_resident_buckets = int(max_resident_buckets)
        self._cache: OrderedDict[tuple, object] = OrderedDict()
        self._cache_lock = threading.Lock()

    @classmethod
    def from_array(
        cls,
        operator,
        chunk_size: int,
        *,
        spill_dir=None,
        max_resident_buckets: int = 2,
        spill_bucket_rows: int | None = None,
    ):
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(
                f"operator must be square, got shape {operator.shape}"
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        H = np.ascontiguousarray(operator, dtype=np.complex128)
        dim = int(H.shape[0])
        if spill_dir is None:
            return cls._from_array_resident(H, chunk_size)
        spill_rows = chunk_size if spill_bucket_rows is None else spill_bucket_rows
        return cls._build_spilled(
            dim=dim,
            chunk_size=chunk_size,
            spill_dir=Path(spill_dir),
            max_resident_buckets=max_resident_buckets,
            dense_operator=H,
            spill_bucket_rows=spill_rows,
        )

    @classmethod
    def from_complex128_file(
        cls,
        path,
        *,
        dim: int,
        chunk_size: int,
        spill_dir=None,
        max_resident_buckets: int = 2,
        data_offset: int = 0,
        spill_bucket_rows: int | None = None,
    ):
        """Row-major ``complex128`` payload of length ``dim*dim``.

        ``data_offset`` skips a leading header (e.g. ``.npy``); the
        payload size check is ``file_size == data_offset + dim*dim*16``.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if data_offset < 0:
            raise ValueError(f"data_offset must be >= 0, got {data_offset}")
        spill_rows = chunk_size if spill_bucket_rows is None else spill_bucket_rows
        path = Path(path)
        payload_bytes = dim * dim * np.dtype(np.complex128).itemsize
        expected_bytes = data_offset + payload_bytes
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path} has {actual_bytes} bytes, expected "
                f"{expected_bytes} for dim={dim} complex128"
                + (f" with data_offset={data_offset}" if data_offset else "")
            )
        if spill_dir is None:
            spill_dir = path.with_suffix(path.suffix + ".buckets")
        spill_dir = Path(spill_dir)

        try:
            dense = np.memmap(
                path,
                dtype=np.complex128,
                mode="r",
                offset=int(data_offset),
                shape=(dim, dim),
                order="C",
            )
        except (ValueError, OSError, TypeError):
            dense = None

        if dense is not None:
            try:
                return cls._build_spilled(
                    dim=dim,
                    chunk_size=chunk_size,
                    spill_dir=spill_dir,
                    max_resident_buckets=max_resident_buckets,
                    dense_operator=dense,
                    spill_bucket_rows=spill_rows,
                )
            finally:
                mmap = getattr(dense, "_mmap", None)
                if mmap is not None:
                    mmap.close()

        row_bytes = dim * np.dtype(np.complex128).itemsize
        with path.open("rb") as fh:

            def row_reader(p: int, *, _fh=fh, _off=data_offset, _rb=row_bytes):
                _fh.seek(_off + p * _rb)
                raw = _fh.read(_rb)
                if len(raw) != _rb:
                    raise ValueError(f"short read for row {p} from {path}")
                return np.frombuffer(raw, dtype=np.complex128).copy()

            return cls._build_spilled(
                dim=dim,
                chunk_size=chunk_size,
                spill_dir=spill_dir,
                max_resident_buckets=max_resident_buckets,
                row_reader=row_reader,
                spill_bucket_rows=spill_rows,
            )

    @classmethod
    def from_dense_file(
        cls,
        path,
        *,
        chunk_size: int,
        meta=None,
        spill_dir=None,
        max_resident_buckets: int = 2,
        spill_bucket_rows: int | None = None,
    ):
        """Build a spilled source from a dense on-disk operator.

        Accepts layout A (raw row-major ``complex128`` + sidecar JSON
        ``paulikit.dense_c128.v1``) or layout B (square C-order
        ``complex128`` ``.npy``). Always spills — never loads the full
        ``dim×dim`` matrix into RAM. Pass the result as
        ``operator_source=`` to ``parallel_decompose_arrays``.

        Parameters
        ----------
        path :
            Operator blob (``.c128`` / raw, or ``.npy``).
        chunk_size :
            Drain tile rows (WHT chunk). Prefer small power-of-two
            values (e.g. 2) for cache locality / parallelism.
        spill_bucket_rows :
            Pass-1 spill tile height. Must be a power of two and a
            multiple of ``chunk_size`` (default: ``chunk_size``).
        meta :
            Sidecar JSON path for layout A (default: ``str(path)+".json"``).
            Optional for ``.npy`` (may carry ``sha256``).
        spill_dir :
            Directory for Pass-1 bucket files (default: ``PATH.buckets``).
            Prefer a fast local volume (Pass-1 is write-bound there).
        max_resident_buckets :
            LRU size for cached drain slices (default 2).

        See also
        --------
        paulikit.algorithms.dense_input.resolve_dense_file
        """
        from paulikit.algorithms.dense_input import resolve_dense_file

        spec = resolve_dense_file(path, meta=meta)
        return cls.from_complex128_file(
            spec.path,
            dim=spec.dim,
            chunk_size=chunk_size,
            spill_dir=spill_dir,
            max_resident_buckets=max_resident_buckets,
            data_offset=spec.data_offset,
            spill_bucket_rows=spill_bucket_rows,
        )

    @classmethod
    def _from_array_resident(cls, H, chunk_size: int):
        dim = int(H.shape[0])
        n_buckets = (dim + chunk_size - 1) // chunk_size
        buckets = [
            np.zeros((_bucket_height(dim, chunk_size, b), dim), dtype=np.complex128)
            for b in range(n_buckets)
        ]
        q_range = np.arange(dim)
        for p in range(dim):
            xs = p ^ q_range
            bucket_ids = xs // chunk_size
            rows = xs - bucket_ids * chunk_size
            values = H[p]
            for b in range(n_buckets):
                mask = bucket_ids == b
                if np.any(mask):
                    buckets[b][rows[mask], q_range[mask]] = values[mask]
        return cls(
            dim=dim,
            chunk_size=chunk_size,
            n_buckets=n_buckets,
            buckets_ram=buckets,
            bucket_files=None,
            spill_dir=None,
            max_resident_buckets=max(n_buckets, 1),
            spill_bucket_rows=chunk_size,
        )

    @classmethod
    def _build_spilled(
        cls,
        *,
        dim: int,
        chunk_size: int,
        spill_dir: Path,
        max_resident_buckets: int,
        row_reader=None,
        dense_operator=None,
        spill_bucket_rows: int | None = None,
    ):
        if row_reader is None and dense_operator is None:
            raise ValueError(
                "_build_spilled requires row_reader or dense_operator"
            )
        spill_rows = chunk_size if spill_bucket_rows is None else int(spill_bucket_rows)
        _validate_spill_vs_chunk(chunk_size, spill_rows, bit_route=True)

        spill_dir.mkdir(parents=True, exist_ok=True)
        n_buckets = (dim + spill_rows - 1) // spill_rows
        bucket_files = [None] * n_buckets
        native = _pass1_native()

        op = dense_operator
        if op is not None and (
            not isinstance(op, np.ndarray)
            or op.dtype != np.complex128
            or not op.flags["C_CONTIGUOUS"]
        ):
            # Never ascontiguousarray a memmap — that densifies H into RAM.
            if not isinstance(op, np.memmap):
                op = np.ascontiguousarray(op, dtype=np.complex128)

        for b in range(n_buckets):
            height = _bucket_height(dim, spill_rows, b)
            x_lo = b * spill_rows
            tile = np.zeros((height, dim), dtype=np.complex128)

            if op is not None and native is not None:
                native.fill_bucket_from_operator(op, x_lo=x_lo, bucket=tile)
            elif op is not None:
                for p in range(dim):
                    _scatter_row_into_bucket_bit(
                        tile, p=p, row=op[p], x_lo=x_lo, height=height
                    )
            else:
                for p in range(dim):
                    row = np.asarray(row_reader(p), dtype=np.complex128)
                    if native is not None:
                        native.scatter_block_into_bucket(
                            row.reshape(1, dim),
                            p_start=p,
                            x_lo=x_lo,
                            bucket=tile,
                        )
                    else:
                        _scatter_row_into_bucket_bit(
                            tile, p=p, row=row, x_lo=x_lo, height=height
                        )

            path = _bucket_path(spill_dir, b)
            _write_bucket(path, tile)
            bucket_files[b] = path
            del tile

        return cls(
            dim=dim,
            chunk_size=chunk_size,
            n_buckets=n_buckets,
            buckets_ram=None,
            bucket_files=bucket_files,
            spill_dir=spill_dir,
            max_resident_buckets=max_resident_buckets,
            spill_bucket_rows=spill_rows,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def spill_bucket_rows(self) -> int:
        return self._spill_bucket_rows

    @property
    def spilled(self) -> bool:
        return self._spill_dir is not None

    def gather_chunk(self, chunk_start: int, n_rows: int):
        if chunk_start % self._chunk_size != 0:
            raise ValueError(
                f"chunk_start={chunk_start} must be a multiple of "
                f"chunk_size={self._chunk_size}"
            )
        if n_rows < 0 or chunk_start < 0 or chunk_start + n_rows > self._dim:
            raise ValueError(
                f"chunk [{chunk_start}, {chunk_start + n_rows}) out of "
                f"range for dim={self._dim}"
            )
        if n_rows == 0:
            return np.empty((0, self._dim), dtype=np.complex128)

        spill_id = chunk_start // self._spill_bucket_rows
        row_in_spill = chunk_start - spill_id * self._spill_bucket_rows
        if spill_id < 0 or spill_id >= self._n_buckets:
            raise ValueError(
                f"chunk_start={chunk_start} out of range for dim={self._dim}"
            )
        height = _bucket_height(self._dim, self._spill_bucket_rows, spill_id)
        if row_in_spill + n_rows > height:
            raise ValueError(
                f"drain chunk [{chunk_start}, {chunk_start + n_rows}) "
                f"crosses spill tile {spill_id} boundary "
                f"(tile rows [{spill_id * self._spill_bucket_rows}, "
                f"{spill_id * self._spill_bucket_rows + height}))"
            )

        ram = self._buckets_ram[spill_id]
        if ram is not None:
            return np.ascontiguousarray(ram[row_in_spill : row_in_spill + n_rows])

        return np.ascontiguousarray(
            self._load_slice(spill_id, height, row_in_spill, n_rows)
        )

    def _load_slice(self, spill_id: int, height: int, row_offset: int, n_rows: int):
        key = (spill_id, row_offset, n_rows)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            path = self._bucket_files[spill_id]
        if path is None:
            raise RuntimeError(f"bucket {spill_id} has no RAM or file backing")

        if row_offset == 0 and n_rows == height:
            slice_ = _read_bucket(Path(path), height, self._dim)
        else:
            slice_ = _read_bucket_slice(Path(path), self._dim, row_offset, n_rows)

        with self._cache_lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = slice_
            while len(self._cache) > self._max_resident_buckets:
                self._cache.popitem(last=False)
            return slice_

    def __getstate__(self):
        return {
            "dim": self._dim,
            "chunk_size": self._chunk_size,
            "spill_bucket_rows": self._spill_bucket_rows,
            "n_buckets": self._n_buckets,
            "buckets_ram": self._buckets_ram,
            "bucket_files": [
                None if p is None else str(p) for p in self._bucket_files
            ],
            "spill_dir": None if self._spill_dir is None else str(self._spill_dir),
            "max_resident_buckets": self._max_resident_buckets,
        }

    def __setstate__(self, state):
        self._dim = int(state["dim"])
        self._chunk_size = int(state["chunk_size"])
        self._spill_bucket_rows = int(
            state.get("spill_bucket_rows", state["chunk_size"])
        )
        self._n_buckets = int(state["n_buckets"])
        self._buckets_ram = list(state["buckets_ram"])
        self._bucket_files = [
            None if p is None else Path(p) for p in state["bucket_files"]
        ]
        spill = state["spill_dir"]
        self._spill_dir = None if spill is None else Path(spill)
        self._max_resident_buckets = int(state["max_resident_buckets"])
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
