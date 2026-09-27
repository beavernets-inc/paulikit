# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""Tests for the CLI's ``--parallel`` decomposition options.

These exercise the command through its real argument parser and entry
point, not by calling internals, because the point is that a user can
actually reach the parallel path from the command line and control it.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import fwht_pauli_coefficients
from paulikit.cli import (
    _default_masses,
    _default_spring_constants,
    build_parser,
    cmd_decompose,
)
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _run(capsys, *argv):
    """Parse argv the way the console script does and run it."""
    args = build_parser().parse_args(["decompose", *argv])
    code = args.func(args)
    return code, capsys.readouterr()


def _expected_term_count(n_oscillators, atol=1e-10):
    h = build_hamiltonian(
        n_oscillators,
        _default_spring_constants(n_oscillators),
        _default_masses(n_oscillators),
        sparse=True,
    )
    padded, _ = pad_to_power_of_two(h, sparse=True)
    _x, _z, c = fwht_pauli_coefficients(
        padded, sparse=True, chunk_size=2, atol=atol
    )
    return len(c)


@pytest.mark.parametrize("executor", ["auto", "thread", "process"])
def test_parallel_reports_the_right_term_count(capsys, executor):
    """Every executor must reach the same answer through the CLI."""
    expected = _expected_term_count(20)
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--parallel", "--executor", executor,
    )
    assert code == 0
    assert f"nonzero Pauli terms: {expected}" in out.out
    assert f"executor={executor}" in out.out


def test_parallel_requires_chunk_size(capsys):
    code, out = _run(capsys, "--n-oscillators", "20", "--parallel")
    assert code == 1
    assert "--chunk-size" in out.err


def test_n_workers_is_honoured_and_does_not_change_the_answer(capsys):
    expected = _expected_term_count(20)
    for workers in ("1", "2"):
        code, out = _run(
            capsys, "--n-oscillators", "20", "--chunk-size", "2",
            "--parallel", "--n-workers", workers,
        )
        assert code == 0
        assert f"nonzero Pauli terms: {expected}" in out.out


def test_show_terms_says_it_is_unavailable_rather_than_ignoring_it(capsys):
    """--show-terms cannot work on a path that never builds labels.
    Silently ignoring a flag the user passed would be worse than
    saying so."""
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--parallel", "--show-terms",
    )
    assert code == 0
    assert "not available with --parallel" in out.out


def test_parallel_builds_the_operator_sparse(capsys, monkeypatch):
    """The whole point of this path is reaching sizes a dense operator
    cannot hold, so it must not densify on the way in."""
    seen = {}
    import paulikit.cli as cli_module

    real = cli_module.build_hamiltonian

    def spy(*a, **kw):
        seen["sparse"] = kw.get("sparse", False)
        return real(*a, **kw)

    monkeypatch.setattr(cli_module, "build_hamiltonian", spy)
    _run(capsys, "--n-oscillators", "20", "--chunk-size", "2", "--parallel")
    assert seen["sparse"] is True

    seen.clear()
    _run(capsys, "--n-oscillators", "20")
    assert seen["sparse"] is False


def test_executor_choices_are_constrained_by_the_parser():
    """An unknown executor must fail at parse time, not at runtime."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["decompose", "--n-oscillators", "20", "--executor", "magic"]
        )


def test_checkpoint_round_trips_through_the_cli(capsys, tmp_path):
    expected = _expected_term_count(20)
    ckpt = tmp_path / "cli.ckpt"
    code, _ = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--parallel", "--checkpoint-path", str(ckpt),
    )
    assert code == 0
    assert ckpt.exists()

    # Resuming from a complete checkpoint must still report the full
    # result rather than nothing.
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--parallel", "--checkpoint-path", str(ckpt),
    )
    assert code == 0
    assert f"nonzero Pauli terms: {expected}" in out.out


def test_progress_is_off_by_default(capsys):
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2", "--parallel",
    )
    assert code == 0
    assert "paulikit: chunks=" not in out.err


def test_progress_prints_on_stderr_for_parallel(capsys):
    expected = _expected_term_count(20)
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--parallel", "--progress",
    )
    assert code == 0
    assert f"nonzero Pauli terms: {expected}" in out.out
    assert "paulikit: chunks=" in out.err
    assert "/" in out.err  # k/N
    assert "%" in out.err
    assert "eta=" in out.err
    assert "terms=" in out.err
    assert "elapsed=" in out.err
    # Progress must not leak onto the normal result stream.
    assert "paulikit: chunks=" not in out.out


def test_progress_prints_on_stderr_for_stream(capsys):
    code, out = _run(
        capsys, "--n-oscillators", "20", "--chunk-size", "2",
        "--stream", "--progress",
    )
    assert code == 0
    assert "paulikit: chunks=" in out.err
    assert "%" in out.err
    assert "eta=" in out.err
    assert "Decomposition time (streamed)" in out.out


@pytest.mark.parametrize("command", [[], ["decompose"], ["benchmark"]])
def test_help_renders(capsys, command):
    """`--help` must actually render.

    argparse runs %-formatting over every help string, so a literal
    percent sign in help text (e.g. "~60% of total time") is read as a
    format specifier and raises TypeError at display time. That is
    invisible to tests which build the parser and parse known argv -
    it only fires when the help is formatted - so this exercises the
    real path a user takes.
    """
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([*command, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip()
