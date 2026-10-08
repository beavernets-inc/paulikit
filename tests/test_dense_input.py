# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Layout A/B detection, sidecar validation, from_dense_file."""

import json
from pathlib import Path

import numpy as np
import pytest

from paulikit.algorithms.dense_bucketed import DenseBucketedSource
from paulikit.algorithms.dense_input import (
    FORMAT_V1,
    resolve_dense_file,
)
from paulikit.algorithms.operator_source import DenseResidentSource


def _random_hermitian(n_qubits: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (raw + raw.conj().T) / 2


def _write_layout_a(tmp_path: Path, operator: np.ndarray, *, bad: dict | None = None):
    path = tmp_path / "H.c128"
    np.ascontiguousarray(operator, dtype=np.complex128).tofile(path)
    dim = operator.shape[0]
    meta = {
        "format": FORMAT_V1,
        "dim": dim,
        "dtype": "complex128",
        "layout": "row-major",
        "endianness": "little",
        "n_qubits": int(dim).bit_length() - 1,
    }
    if bad:
        meta.update(bad)
    meta_path = Path(str(path) + ".json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return path, meta_path


def _assert_matches_resident(operator: np.ndarray, bucketed: DenseBucketedSource):
    dim = operator.shape[0]
    resident = DenseResidentSource(operator)
    for chunk_start in range(0, dim, bucketed.chunk_size):
        n_rows = min(bucketed.chunk_size, dim - chunk_start)
        a = resident.gather_chunk(chunk_start, n_rows)
        b = bucketed.gather_chunk(chunk_start, n_rows)
        np.testing.assert_array_equal(a, b)


def test_resolve_layout_a(tmp_path: Path):
    op = _random_hermitian(3, seed=1)
    path, meta_path = _write_layout_a(tmp_path, op)
    spec = resolve_dense_file(path)
    assert spec.layout == "raw_c128"
    assert spec.dim == 8
    assert spec.data_offset == 0
    assert spec.meta_path == meta_path
    assert spec.codec == "identity"


def test_resolve_layout_b_npy(tmp_path: Path):
    op = _random_hermitian(3, seed=2)
    path = tmp_path / "H.npy"
    np.save(path, np.ascontiguousarray(op, dtype=np.complex128))
    spec = resolve_dense_file(path)
    assert spec.layout == "npy"
    assert spec.dim == 8
    assert spec.data_offset > 0


@pytest.mark.parametrize("n_qubits", [3, 4])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_from_dense_file_raw_bit_identical(tmp_path: Path, n_qubits, chunk_size):
    op = _random_hermitian(n_qubits, seed=10 + n_qubits + chunk_size)
    path, _ = _write_layout_a(tmp_path, op)
    bucketed = DenseBucketedSource.from_dense_file(
        path,
        chunk_size=chunk_size,
        spill_dir=tmp_path / "buckets",
        max_resident_buckets=1,
    )
    _assert_matches_resident(op, bucketed)


@pytest.mark.parametrize("n_qubits", [3, 4])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_from_dense_file_npy_bit_identical(tmp_path: Path, n_qubits, chunk_size):
    op = _random_hermitian(n_qubits, seed=20 + n_qubits + chunk_size)
    path = tmp_path / "H.npy"
    np.save(path, np.ascontiguousarray(op, dtype=np.complex128))
    bucketed = DenseBucketedSource.from_dense_file(
        path,
        chunk_size=chunk_size,
        spill_dir=tmp_path / "npy_buckets",
        max_resident_buckets=1,
    )
    _assert_matches_resident(op, bucketed)


def test_sidecar_rejects_bad_format(tmp_path: Path):
    op = _random_hermitian(2, seed=3)
    path, _ = _write_layout_a(tmp_path, op, bad={"format": "nope"})
    with pytest.raises(ValueError, match="format must be"):
        resolve_dense_file(path)


def test_sidecar_rejects_non_power_of_two_dim(tmp_path: Path):
    path = tmp_path / "H.c128"
    # 9x9 is not power of two; write matching bytes then lie in meta? 
    # Better: write 8x8 but claim dim=6.
    op = _random_hermitian(3, seed=4)
    np.ascontiguousarray(op, dtype=np.complex128).tofile(path)
    meta_path = Path(str(path) + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "format": FORMAT_V1,
                "dim": 6,
                "dtype": "complex128",
                "layout": "row-major",
                "endianness": "little",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="power of two"):
        resolve_dense_file(path)


def test_sidecar_rejects_size_mismatch(tmp_path: Path):
    path = tmp_path / "H.c128"
    path.write_bytes(b"\x00" * 32)
    meta_path = Path(str(path) + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "format": FORMAT_V1,
                "dim": 8,
                "dtype": "complex128",
                "layout": "row-major",
                "endianness": "little",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected"):
        resolve_dense_file(path)


def test_npy_rejects_real_dtype(tmp_path: Path):
    path = tmp_path / "H.npy"
    np.save(path, np.eye(4, dtype=np.float64))
    with pytest.raises(ValueError, match="complex128"):
        resolve_dense_file(path)


def test_missing_sidecar(tmp_path: Path):
    path = tmp_path / "H.c128"
    np.eye(4, dtype=np.complex128).tofile(path)
    with pytest.raises(FileNotFoundError, match="sidecar"):
        resolve_dense_file(path)
