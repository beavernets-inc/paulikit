# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Dense operator file layouts for out-of-core input (A0/A1).

Accepted layouts (identity codec only in v1):

* **A** — raw row-major ``complex128`` blob + sidecar JSON
  (``paulikit.dense_c128.v1``)
* **B** — square C-order ``complex128`` ``.npy``

Compression codecs are reserved via ``codec="identity"``; streamed
zstd/FPC may plug in later without changing the gather contract.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

FORMAT_V1 = "paulikit.dense_c128.v1"
CODEC_IDENTITY = "identity"

LayoutName = Literal["raw_c128", "npy"]


@dataclass(frozen=True)
class DenseFileSpec:
    """Resolved on-disk dense operator after validation."""

    path: Path
    dim: int
    layout: LayoutName
    data_offset: int
    codec: str = CODEC_IDENTITY
    meta_path: Path | None = None
    n_qubits: int | None = None


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _host_endianness() -> str:
    return "little" if sys.byteorder == "little" else "big"


def default_meta_path(path: Path) -> Path:
    """Companion sidecar: ``H.c128`` → ``H.c128.json``."""
    return Path(str(path) + ".json")


def load_sidecar(meta_path: Path) -> dict:
    with Path(meta_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"{meta_path}: sidecar must be a JSON object, "
            f"got {type(data).__name__}"
        )
    return data


def validate_sidecar(meta: dict, *, path: Path, meta_path: Path) -> tuple[int, int | None]:
    """Return ``(dim, n_qubits_or_None)`` after checking required fields."""
    fmt = meta.get("format")
    if fmt != FORMAT_V1:
        raise ValueError(
            f"{meta_path}: format must be {FORMAT_V1!r}, got {fmt!r}"
        )
    try:
        dim = int(meta["dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{meta_path}: required integer field 'dim' missing or invalid"
        ) from exc
    if dim < 1 or not _is_power_of_two(dim):
        raise ValueError(
            f"{meta_path}: dim must be a power of two >= 1, got {dim}"
        )

    dtype = meta.get("dtype")
    if dtype != "complex128":
        raise ValueError(
            f"{meta_path}: dtype must be 'complex128' in v1, got {dtype!r}"
        )
    layout = meta.get("layout")
    if layout != "row-major":
        raise ValueError(
            f"{meta_path}: layout must be 'row-major' in v1, got {layout!r}"
        )
    endianness = meta.get("endianness")
    if endianness not in ("little", "big"):
        raise ValueError(
            f"{meta_path}: endianness must be 'little' or 'big', "
            f"got {endianness!r}"
        )
    host = _host_endianness()
    if endianness != host:
        raise ValueError(
            f"{meta_path}: endianness={endianness!r} does not match "
            f"host {host!r} (v1 refuses silent byte-swap)"
        )

    n_qubits = meta.get("n_qubits")
    if n_qubits is not None:
        n_qubits = int(n_qubits)
        if 2**n_qubits != dim:
            raise ValueError(
                f"{meta_path}: n_qubits={n_qubits} does not match "
                f"dim={dim} (expected log2(dim)={int(dim).bit_length() - 1})"
            )

    codec = meta.get("codec", CODEC_IDENTITY)
    if codec != CODEC_IDENTITY:
        raise ValueError(
            f"{meta_path}: codec={codec!r} not supported in v1 "
            f"(only {CODEC_IDENTITY!r})"
        )

    expected_bytes = dim * dim * np.dtype(np.complex128).itemsize
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path}: size {actual_bytes} bytes, expected "
            f"{expected_bytes} for dim={dim} complex128"
        )

    sha = meta.get("sha256")
    if sha is not None:
        import hashlib

        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        got = digest.hexdigest()
        if got.lower() != str(sha).lower():
            raise ValueError(
                f"{path}: sha256 mismatch: sidecar has {sha}, file has {got}"
            )

    return dim, n_qubits


def _read_npy_header(path: Path) -> tuple[int, int]:
    """Return ``(dim, data_offset)`` for a square C-order complex128 .npy."""
    with Path(path).open("rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(
                f"{path}: unsupported .npy version {version}"
            )
        data_offset = fh.tell()

    if fortran_order:
        raise ValueError(
            f"{path}: Fortran-order .npy not accepted in v1 "
            "(need C-contiguous / row-major)"
        )
    if dtype != np.dtype(np.complex128):
        raise ValueError(
            f"{path}: dtype must be complex128, got {dtype}"
        )
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"{path}: array must be square 2-D, got shape {shape}"
        )
    dim = int(shape[0])
    if not _is_power_of_two(dim):
        raise ValueError(
            f"{path}: dim must be a power of two, got {dim}"
        )
    expected = data_offset + dim * dim * np.dtype(np.complex128).itemsize
    actual = Path(path).stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: size {actual} bytes, expected {expected} "
            f"for shape ({dim},{dim}) complex128 with header"
        )
    return dim, data_offset


def resolve_dense_file(path, meta=None) -> DenseFileSpec:
    """Detect layout A/B, validate metadata or ``.npy`` header, return a spec.

    Parameters
    ----------
    path :
        Path to the raw ``complex128`` blob or to a ``.npy`` file.
    meta :
        Optional sidecar JSON. Required for raw layout A when the
        default ``str(path)+".json"`` companion is absent. Optional for
        ``.npy`` (may carry ``sha256`` / ``n_qubits``).

    Returns
    -------
    DenseFileSpec
        Resolved ``dim``, ``layout``, ``data_offset``, and paths.

    Raises
    ------
    FileNotFoundError
        Missing operator or required sidecar.
    ValueError
        Invalid format, dtype, endianness, size, or non-power-of-two dim.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"operator file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        if meta is not None:
            # Optional sidecar may carry sha256 only; dim comes from header.
            meta_path = Path(meta)
            side = load_sidecar(meta_path)
            codec = side.get("codec", CODEC_IDENTITY)
            if codec != CODEC_IDENTITY:
                raise ValueError(
                    f"{meta_path}: codec={codec!r} not supported in v1"
                )
            sha = side.get("sha256")
            if sha is not None:
                import hashlib

                digest = hashlib.sha256()
                with path.open("rb") as fh:
                    for block in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(block)
                got = digest.hexdigest()
                if got.lower() != str(sha).lower():
                    raise ValueError(
                        f"{path}: sha256 mismatch: sidecar has {sha}, "
                        f"file has {got}"
                    )
            dim, data_offset = _read_npy_header(path)
            n_qubits = side.get("n_qubits")
            if n_qubits is not None:
                n_qubits = int(n_qubits)
                if 2**n_qubits != dim:
                    raise ValueError(
                        f"{meta_path}: n_qubits={n_qubits} != log2(dim) "
                        f"for dim={dim}"
                    )
            return DenseFileSpec(
                path=path,
                dim=dim,
                layout="npy",
                data_offset=data_offset,
                codec=CODEC_IDENTITY,
                meta_path=meta_path,
                n_qubits=n_qubits,
            )
        dim, data_offset = _read_npy_header(path)
        return DenseFileSpec(
            path=path,
            dim=dim,
            layout="npy",
            data_offset=data_offset,
            codec=CODEC_IDENTITY,
            meta_path=None,
            n_qubits=int(dim).bit_length() - 1,
        )

    # Layout A: raw + sidecar
    if meta is None:
        meta_path = default_meta_path(path)
    else:
        meta_path = Path(meta)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"sidecar metadata not found: {meta_path} "
            f"(required for raw complex128 layout)"
        )
    side = load_sidecar(meta_path)
    dim, n_qubits = validate_sidecar(side, path=path, meta_path=meta_path)
    return DenseFileSpec(
        path=path,
        dim=dim,
        layout="raw_c128",
        data_offset=0,
        codec=side.get("codec", CODEC_IDENTITY),
        meta_path=meta_path,
        n_qubits=n_qubits if n_qubits is not None else int(dim).bit_length() - 1,
    )
