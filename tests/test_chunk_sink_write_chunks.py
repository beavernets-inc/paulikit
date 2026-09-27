# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Tests for the main-thread chunk sink and CLI --write-chunks."""

import numpy as np

from paulikit.algorithms.fwht import (
    CheckpointChunkSink,
    NullChunkSink,
    _make_chunk_sink,
    iter_checkpoint_chunks,
    parallel_decompose_arrays,
)
from paulikit.cli import (
    _default_masses,
    _default_spring_constants,
    build_parser,
    cmd_decompose,
)
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def test_null_sink_is_the_default_for_none_path():
    sink = _make_chunk_sink(None)
    assert isinstance(sink, NullChunkSink)
    # Must accept the same call shape as the real sink (branchless drain).
    sink.write_parallel_chunk(
        set(), 0,
        np.zeros(0, dtype=np.uint32),
        np.zeros(0, dtype=np.uint32),
        np.zeros(0, dtype=complex),
        np.dtype(np.uint32),
    )


def test_make_chunk_sink_opens_checkpoint_sink_for_a_path(tmp_path):
    sink = _make_chunk_sink(tmp_path / "out.pkcp")
    assert isinstance(sink, CheckpointChunkSink)


def test_write_chunks_cli_persists_readable_frames(capsys, tmp_path):
    out = tmp_path / "cli_chunks.pkcp"
    args = build_parser().parse_args([
        "decompose",
        "--n-oscillators", "20",
        "--chunk-size", "2",
        "--parallel",
        "--executor", "thread",
        "--write-chunks", str(out),
    ])
    assert cmd_decompose(args) == 0
    captured = capsys.readouterr()
    assert "Writing streamed chunks" in captured.out
    assert out.exists()

    frames = list(iter_checkpoint_chunks(out))
    assert frames
    total = sum(len(coeff) for _i, _x, _z, coeff in frames)
    assert f"nonzero Pauli terms: {total}" in captured.out


def test_write_chunks_and_checkpoint_path_must_agree(capsys, tmp_path):
    args = build_parser().parse_args([
        "decompose",
        "--n-oscillators", "8",
        "--chunk-size", "2",
        "--parallel",
        "--write-chunks", str(tmp_path / "a.pkcp"),
        "--checkpoint-path", str(tmp_path / "b.pkcp"),
    ])
    assert cmd_decompose(args) == 1
    assert "must name the same path" in capsys.readouterr().err


def test_library_checkpoint_path_round_trips_via_iter_checkpoint_chunks(tmp_path):
    n = 20
    h = build_hamiltonian(
        n, _default_spring_constants(n), _default_masses(n), sparse=True,
    )
    padded, _nq = pad_to_power_of_two(h, sparse=True)
    path = tmp_path / "lib.pkcp"
    live_terms = 0
    for x, z, coeff in parallel_decompose_arrays(
        padded, chunk_size=2, checkpoint_path=path, executor="thread",
    ):
        live_terms += len(coeff)
    file_terms = sum(
        len(coeff) for _i, _x, _z, coeff in iter_checkpoint_chunks(path)
    )
    assert file_terms == live_terms
    assert live_terms > 0
