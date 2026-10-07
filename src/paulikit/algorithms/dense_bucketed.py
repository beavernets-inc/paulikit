# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Bucketed / spill / file-backed dense OperatorSource (opt-in).

Import this module only when constructing a bucketed dense source —
not from ``fwht``'s default import path. Stdlib ``pathlib`` /
``collections`` stay here so the resident/sparse hot path in
``operator_source`` does not pay for them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _bucket_height(dim: int, chunk_size: int, bucket_id: int) -> int:
    return min(chunk_size, dim - bucket_id * chunk_size)


def _bucket_path(spill_dir: Path, bucket_id: int) -> Path:
    return spill_dir / f"bucket_{bucket_id:06d}.c128"


def _write_bucket(path: Path, bucket) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(bucket, dtype=np.complex128).tofile(path)


def _read_bucket(path: Path, n_rows: int, dim: int):
    expected = n_rows * dim
    flat = np.fromfile(path, dtype=np.complex128, count=expected)
    if flat.size != expected:
        raise ValueError(
            f"bucket file {path} has {flat.size} complex128 values, "
            f"expected {expected} ({n_rows}×{dim})"
        )
    return np.ascontiguousarray(flat.reshape(n_rows, dim))


def _scatter_row_into_bucket(
    bucket,
    *,
    p: int,
    row,
    q_range,
    x_lo: int,
    x_hi: int,
) -> None:
    xs = p ^ q_range
    mask = (xs >= x_lo) & (xs < x_hi)
    if np.any(mask):
        bucket[xs[mask] - x_lo, q_range[mask]] = row[mask]


class DenseBucketedSource:
    """Dense gather via scatter into fixed-size x-buckets.

    Without ``spill_dir``, all buckets stay in RAM. With ``spill_dir``,
    build is one-bucket-at-a-time (peak ≈ one bucket + one operator row)
    and gathers reload from raw ``complex128`` tiles with a small LRU.
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
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if max_resident_buckets < 1:
            raise ValueError(
                f"max_resident_buckets must be >= 1, got {max_resident_buckets}"
            )
        self._dim = int(dim)
        self._chunk_size = int(chunk_size)
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
        self._cache: OrderedDict[int, object] = OrderedDict()
        self._cache_lock = threading.Lock()

    @classmethod
    def from_array(
        cls,
        operator,
        chunk_size: int,
        *,
        spill_dir=None,
        max_resident_buckets: int = 2,
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
        return cls._build_spilled(
            dim=dim,
            chunk_size=chunk_size,
            spill_dir=Path(spill_dir),
            max_resident_buckets=max_resident_buckets,
            row_reader=lambda p: H[p],
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
    ):
        """Row-major ``complex128`` file of length ``dim*dim``."""
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        path = Path(path)
        expected_bytes = dim * dim * np.dtype(np.complex128).itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path} has {actual_bytes} bytes, expected "
                f"{expected_bytes} for dim={dim} complex128"
            )
        if spill_dir is None:
            spill_dir = path.with_suffix(path.suffix + ".buckets")
        spill_dir = Path(spill_dir)
        row_bytes = dim * np.dtype(np.complex128).itemsize

        def row_reader(p: int):
            with path.open("rb") as fh:
                fh.seek(p * row_bytes)
                raw = fh.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f"short read for row {p} from {path}")
            return np.frombuffer(raw, dtype=np.complex128).copy()

        return cls._build_spilled(
            dim=dim,
            chunk_size=chunk_size,
            spill_dir=spill_dir,
            max_resident_buckets=max_resident_buckets,
            row_reader=row_reader,
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
        )

    @classmethod
    def _build_spilled(
        cls,
        *,
        dim: int,
        chunk_size: int,
        spill_dir: Path,
        max_resident_buckets: int,
        row_reader,
    ):
        spill_dir.mkdir(parents=True, exist_ok=True)
        n_buckets = (dim + chunk_size - 1) // chunk_size
        q_range = np.arange(dim)
        bucket_files = [None] * n_buckets
        for b in range(n_buckets):
            height = _bucket_height(dim, chunk_size, b)
            bucket = np.zeros((height, dim), dtype=np.complex128)
            x_lo = b * chunk_size
            x_hi = x_lo + height
            for p in range(dim):
                _scatter_row_into_bucket(
                    bucket,
                    p=p,
                    row=np.asarray(row_reader(p), dtype=np.complex128),
                    q_range=q_range,
                    x_lo=x_lo,
                    x_hi=x_hi,
                )
            path = _bucket_path(spill_dir, b)
            _write_bucket(path, bucket)
            bucket_files[b] = path
            del bucket
        return cls(
            dim=dim,
            chunk_size=chunk_size,
            n_buckets=n_buckets,
            buckets_ram=None,
            bucket_files=bucket_files,
            spill_dir=spill_dir,
            max_resident_buckets=max_resident_buckets,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def spilled(self) -> bool:
        return self._spill_dir is not None

    def gather_chunk(self, chunk_start: int, n_rows: int):
        if chunk_start % self._chunk_size != 0:
            raise ValueError(
                f"chunk_start={chunk_start} must be a multiple of "
                f"chunk_size={self._chunk_size}"
            )
        bucket_id = chunk_start // self._chunk_size
        if bucket_id < 0 or bucket_id >= self._n_buckets:
            raise ValueError(
                f"chunk_start={chunk_start} out of range for dim={self._dim}"
            )
        height = _bucket_height(self._dim, self._chunk_size, bucket_id)
        if n_rows != height:
            raise ValueError(
                f"n_rows={n_rows} does not match bucket "
                f"{bucket_id} height {height}"
            )
        return np.ascontiguousarray(self._load_bucket(bucket_id, height))

    def _load_bucket(self, bucket_id: int, height: int):
        ram = self._buckets_ram[bucket_id]
        if ram is not None:
            return ram
        # ThreadPoolExecutor shares one source; protect the LRU.
        # File I/O stays outside the lock so concurrent misses can
        # read different buckets in parallel.
        with self._cache_lock:
            if bucket_id in self._cache:
                self._cache.move_to_end(bucket_id)
                return self._cache[bucket_id]
            path = self._bucket_files[bucket_id]
        if path is None:
            raise RuntimeError(f"bucket {bucket_id} has no RAM or file backing")
        bucket = _read_bucket(Path(path), height, self._dim)
        with self._cache_lock:
            existing = self._cache.get(bucket_id)
            if existing is not None:
                self._cache.move_to_end(bucket_id)
                return existing
            self._cache[bucket_id] = bucket
            while len(self._cache) > self._max_resident_buckets:
                self._cache.popitem(last=False)
            return bucket

    def __getstate__(self):
        return {
            "dim": self._dim,
            "chunk_size": self._chunk_size,
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
