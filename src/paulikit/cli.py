#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Command-line interface for the paulikit package.

The other modules (``paulikit.hamiltonian``, ``paulikit.pauli_utils``,
``paulikit.testing.fixtures``, ``paulikit.algorithms.fwht``) are plain
importable library code with no CLI of their own by design - keeping
them free of argument-parsing concerns makes them easier to unit test
and reuse as a library. This module is the one place that wires them
together into runnable subcommands.

Usage
-----
Once installed (``pip install -e . --no-build-isolation`` from the
package root, or ``pip install paulikit`` once published), the
``paulikit`` console script is available directly::

    paulikit --help
    paulikit decompose --help
    paulikit decompose --n-oscillators 4
    paulikit benchmark --n-oscillators 16 30 50
    paulikit regenerate-fixtures

Without installing, it can also be run as a module from the package's
``src/`` directory::

    python -m paulikit.cli --help

See README.md for example invocations, or run
``paulikit <subcommand> --help`` for what each subcommand does.
"""

import argparse
import sys
import time

from paulikit import __version__
from paulikit.algorithms.fwht import (
    fwht_pauli_coefficients,
    fwht_pauli_terms,
    fwht_pauli_terms_iter,
)
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _default_spring_constants(n_oscillators):
    """A simple, deterministic (not physically meaningful) parameter
    set used by the decompose/benchmark subcommands when the caller
    doesn't need a specific physical system - just something concrete
    and reproducible to decompose. Scales with N so larger N doesn't
    degenerate into all-zero or all-equal matrices."""
    constants = {}
    for i in range(n_oscillators):
        for j in range(i, n_oscillators):
            constants[(i, j)] = 1.0 + 0.1 * (i + j)
    return constants


def _default_masses(n_oscillators):
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


def _resolve_chunk_write_path(args):
    """Resolve CLI stream-writer path (--write-chunks / --checkpoint-path).

    Both flags name the same main-thread PKCP sink. Prefer
    ``--write-chunks`` in new docs; ``--checkpoint-path`` remains as the
    historical alias (and still enables crash/resume).
    """
    write = getattr(args, "write_chunks", None)
    ckpt = getattr(args, "checkpoint_path", None)
    if write is not None and ckpt is not None and write != ckpt:
        print(
            "paulikit: --write-chunks and --checkpoint-path must name "
            "the same path when both are set "
            f"(got {write!r} and {ckpt!r})",
            file=sys.stderr,
        )
        return None, 1
    return write or ckpt, 0


def cmd_decompose(args):
    """Build a synthetic N-oscillator Hamiltonian and Pauli-decompose it."""
    n = args.n_oscillators
    spring_constants = _default_spring_constants(n)
    masses = _default_masses(n)

    write_path, err = _resolve_chunk_write_path(args)
    if err:
        return err

    # --parallel builds the operator SPARSE. That is not an
    # optimisation detail: at 15 qubits a dense operator is 16 GiB and
    # at 16 qubits it is 64 GiB, so densifying here would put the
    # sizes this path exists to reach out of reach before the
    # decomposition even starts.
    sparse_input = bool(getattr(args, "parallel", False))
    unpadded = build_hamiltonian(n, spring_constants, masses,
                                 sparse=sparse_input)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=sparse_input)

    print(f"N={n} oscillators, {n_qubits} qubits, {padded.shape[0]}x{padded.shape[0]} "
          f"padded Hamiltonian")
    if write_path is not None:
        print(
            f"Writing streamed chunks to {write_path} "
            f"(symplectic x, z, coeff; resume-capable PKCP frames)"
        )

    if getattr(args, "parallel", False):
        if not args.chunk_size:
            print("--parallel requires --chunk-size", file=sys.stderr)
            return 1

        from paulikit.algorithms.fwht import parallel_decompose_arrays

        start = time.perf_counter()
        total_terms = 0
        n_chunks = 0
        for _x, _z, coeff in parallel_decompose_arrays(
            padded,
            chunk_size=args.chunk_size,
            n_workers=args.n_workers,
            atol=args.atol,
            checkpoint_path=write_path,
            executor=args.executor,
            eager_threads=args.eager_threads,
        ):
            n_chunks += 1
            total_terms += len(coeff)
        elapsed = time.perf_counter() - start

        print(f"Decomposition time (parallel, executor={args.executor}): "
              f"{elapsed:.4f}s")
        print(f"Chunks: {n_chunks}, nonzero Pauli terms: {total_terms}")
        if args.show_terms:
            # Labels are deliberately not built on this path - that is
            # the serial cost it exists to avoid. Say so rather than
            # silently ignoring the flag.
            print("(--show-terms not available with --parallel: this path "
                  "yields raw arrays and never builds labels; use "
                  "terms_from_arrays on the chunks you actually need)")
        return 0

    if args.stream:
        # Exercises fwht_pauli_terms_iter directly: yields one dict
        # per chunk instead of building one combined dict for the
        # whole operator - see that function's docstring for why this
        # is a real divide-and-conquer decomposition, not just a
        # memory workaround. --show-terms prints each chunk's
        # terms as they arrive, rather than sorting the full combined
        # set at the end (which would defeat the point at large N).
        if not args.chunk_size:
            print("--stream requires --chunk-size (no whole-array streaming mode)",
                  file=sys.stderr)
            return 1

        start = time.perf_counter()
        total_terms = 0
        n_chunks = 0
        for chunk_terms in fwht_pauli_terms_iter(
            padded,
            chunk_size=args.chunk_size,
            atol=args.atol,
            checkpoint_path=write_path,
            parallel_labels=args.parallel_labels,
        ):
            n_chunks += 1
            total_terms += len(chunk_terms)
            if args.show_terms:
                for label in sorted(chunk_terms):
                    print(f"  {label}: {chunk_terms[label]!r}")
        elapsed = time.perf_counter() - start

        print(f"Decomposition time (streamed): {elapsed:.4f}s")
        print(f"Chunks: {n_chunks}, nonzero Pauli terms: {total_terms}")
        return 0

    if args.sparse_output:
        # Exercises fwht_pauli_coefficients(sparse=True) directly. With
        # no --chunk-size, returns the dense-block (active_x,
        # active_coefficients) form; with --chunk-size, returns the
        # already-thresholded COO (x, z, coefficient) triple form
        # instead - this flag is for inspecting/timing that raw
        # output shape, not a switch on
        # fwht_pauli_terms's own algorithm.
        start = time.perf_counter()
        result = fwht_pauli_coefficients(
            padded,
            sparse=True,
            chunk_size=args.chunk_size,
            atol=args.atol,
            checkpoint_path=write_path,
        )
        elapsed = time.perf_counter() - start

        if args.chunk_size is not None:
            x_out, z_out, coeff_out = result
            print(f"Decomposition time (sparse output, chunked): {elapsed:.4f}s")
            print(f"Nonzero terms: {len(x_out)}")
        else:
            active_x, active_coefficients = result
            print(f"Decomposition time (sparse output): {elapsed:.4f}s")
            print(f"Active rows: {len(active_x)} of {padded.shape[0]} possible")

        if args.show_terms:
            terms = fwht_pauli_terms(
                padded,
                atol=args.atol,
                chunk_size=args.chunk_size,
                checkpoint_path=write_path,
            )
            for label in sorted(terms):
                print(f"  {label}: {terms[label]!r}")
        return 0

    start = time.perf_counter()
    terms = fwht_pauli_terms(
        padded,
        atol=args.atol,
        chunk_size=args.chunk_size,
        checkpoint_path=write_path,
    )
    elapsed = time.perf_counter() - start

    print(f"Decomposition time: {elapsed:.4f}s")
    print(f"Nonzero Pauli terms: {len(terms)}")

    if args.show_terms:
        for label in sorted(terms):
            print(f"  {label}: {terms[label]!r}")

    return 0


def cmd_benchmark(args):
    """Time the FWHT decomposition across a sweep of N (oscillator count) values."""
    if args.compare_dense_sparse:
        print(f"{'N':>5} {'qubits':>7} {'dim':>6} {'active':>8} "
              f"{'dense (s)':>10} {'sparse (s)':>11}")
        for n in args.n_oscillators:
            spring_constants = _default_spring_constants(n)
            masses = _default_masses(n)
            unpadded = build_hamiltonian(n, spring_constants, masses)
            padded, n_qubits = pad_to_power_of_two(unpadded)

            start = time.perf_counter()
            dense_coefficients = fwht_pauli_coefficients(padded, sparse=False)
            dense_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            active_x, _ = fwht_pauli_coefficients(padded, sparse=True)
            sparse_elapsed = time.perf_counter() - start

            print(f"{n:>5} {n_qubits:>7} {padded.shape[0]:>6} {len(active_x):>8} "
                  f"{dense_elapsed:>10.4f} {sparse_elapsed:>11.4f}")
            del dense_coefficients

        return 0

    print(f"{'N':>5} {'qubits':>7} {'dim':>6} {'terms':>8} {'time (s)':>10}")
    for n in args.n_oscillators:
        spring_constants = _default_spring_constants(n)
        masses = _default_masses(n)
        unpadded = build_hamiltonian(n, spring_constants, masses)
        padded, n_qubits = pad_to_power_of_two(unpadded)

        start = time.perf_counter()
        terms = fwht_pauli_terms(padded, atol=args.atol, chunk_size=args.chunk_size)
        elapsed = time.perf_counter() - start

        print(f"{n:>5} {n_qubits:>7} {padded.shape[0]:>6} {len(terms):>8} {elapsed:>10.4f}")

    return 0


def cmd_regenerate_fixtures(args):
    """Regenerate paulikit.testing.fixtures's expected_terms constants.

    Thin wrapper around ``paulikit.testing.fixtures.generate_fixture_data``;
    kept here so it's discoverable alongside the other subcommands and
    documented in one place (--help).
    """
    from paulikit.testing.fixtures import generate_fixture_data

    try:
        generate_fixture_data()
    except ImportError:
        # PennyLane is the oracle these constants are generated from,
        # and it is a development-only dependency - absent in a normal
        # install, which is the point. Say so rather than showing an
        # import traceback for an optional tool.
        print(
            "paulikit: regenerating fixtures requires PennyLane, which "
            "is a development-only dependency and is not installed. "
            "Install it with 'pip install pennylane' to run this "
            "command.",
            file=sys.stderr,
        )
        return 1
    print(
        "\nReminder: this prints regenerated constants but does not "
        "edit fixtures.py automatically. Review the output and paste "
        "into paulikit/testing/fixtures.py's FIXTURE_N2/FIXTURE_N4 "
        "definitions by hand if they should change - see that module's "
        "docstring.",
        file=sys.stderr,
    )
    return 0


def _positive_int(value: str) -> int:
    """argparse ``type`` for an oscillator count.

    Rejected at parse time rather than deeper in the stack: N <= 0
    otherwise reaches log2(0) during padding and surfaces as an
    OverflowError traceback, which tells the user nothing about which
    argument was wrong.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {number}"
        )
    return number


def build_parser():
    """Construct the top-level argparse.ArgumentParser for the paulikit CLI."""
    parser = argparse.ArgumentParser(
        prog="paulikit",
        description=(
            "Exact Pauli decomposition of complex matrices, with "
            "streamed output so peak memory is bounded by chunk size "
            "rather than term count."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Build a synthetic N-oscillator Hamiltonian and decompose it",
        description=(
            "Builds a coupled-oscillator Hamiltonian for a given N using "
            "a fixed, deterministic (not physically calibrated) set of "
            "spring constants and masses, pads it to a power-of-two "
            "dimension, and runs the FWHT-based Pauli decomposition on "
            "it, reporting timing and term count."
        ),
    )
    decompose_parser.add_argument(
        "--n-oscillators", "-n", type=_positive_int, default=2,
        help="Number of coupled oscillators, N (default: 2)",
    )
    decompose_parser.add_argument(
        "--atol", type=float, default=1e-10,
        help="Absolute coefficient threshold below which a term is "
             "dropped as zero (default: 1e-10)",
    )
    decompose_parser.add_argument(
        "--show-terms", action="store_true",
        help="Print every nonzero Pauli term and its coefficient "
             "(omitted by default since term counts grow quickly with N)",
    )
    decompose_parser.add_argument(
        "--sparse-output", action="store_true",
        help="Call fwht_pauli_coefficients(sparse=True) directly and "
             "report its timing/active-row count, instead of the usual "
             "label->coefficient dict via fwht_pauli_terms (which "
             "already uses the sparse path internally either way - "
             "this flag is for inspecting the raw sparse output shape "
             "itself)",
    )
    decompose_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Process active rows in blocks of at most this size instead "
             "of one (n_active, dim) array all at once, bounding peak "
             "memory to roughly chunk_size * dim complex entries - needed "
             "at large N where the whole-array approach exhausts memory "
             "(default: None). On the library APIs "
             "(parallel_decompose / parallel_decompose_arrays) omitting "
             "chunk_size auto-tunes against measured cache boundaries; "
             "this CLI flag is always explicit. Required with --stream "
             "and with --parallel.",
    )
    decompose_parser.add_argument(
        "--stream", action="store_true",
        help="Use fwht_pauli_terms_iter instead of fwht_pauli_terms: "
             "yields one label->coefficient dict per chunk instead of "
             "building one combined dict for the whole operator, so peak "
             "memory never depends on the total term count - needed at "
             "N where the full result (e.g. 91.65M terms at N=150) does "
             "not fit in memory even after the accumulator fix. "
             "Requires --chunk-size.",
    )
    decompose_parser.add_argument(
        "--parallel-labels", action="store_true",
        help="With --stream, use the oneTBB-parallel label kernel "
             "instead of the serial one for each chunk. Wins ~1.1-1.4x "
             "wall-clock in isolation, but delivers no measurable "
             "benefit once embedded in the real streaming pipeline at "
             "N=150 (dict construction there dominates at ~60%% of "
             "total time, dwarfing labeling's ~7%% share). Ignored "
             "without --stream.",
    )
    decompose_parser.add_argument(
        "--parallel", action="store_true",
        help="Decompose across multiple workers via "
             "parallel_decompose_arrays, streaming raw (x, z, coeff) "
             "arrays per chunk rather than building labels. This is "
             "the path that scales: at N=150 it does 91.6M terms in "
             "~1.0s in ~72 MiB, and it reaches problem sizes a dense "
             "implementation cannot hold at all (15 qubits needs "
             "16 GiB dense, 16 qubits needs 64 GiB). Requires "
             "--chunk-size. Labels are not built, so --show-terms "
             "reports counts only.",
    )
    decompose_parser.add_argument(
        "--executor", choices=("auto", "thread", "process"), default="auto",
        help="With --parallel, how to drain chunks. 'thread' runs the "
             "compiled kernels concurrently with no pickling (both "
             "release the GIL); 'process' uses a process pool, which "
             "pays IPC but does not depend on the kernels being built. "
             "'auto' (default) picks thread when the compiled kernels "
             "are available and process otherwise - the right choice "
             "differs by ~6x in each direction, so it is decided per "
             "build rather than globally.",
    )
    decompose_parser.add_argument(
        "--n-workers", type=int, default=None,
        help="With --parallel, how many workers. Defaults to the "
             "number of distinct physical cores (not logical CPUs - "
             "hyperthread siblings share execution units and measured "
             "worse). On the 4-core development machine, 4 threads "
             "measured 3.44x against an Amdahl ceiling of 3.63x, "
             "while 8 cost 1.72x the cycles for no wall-clock gain.",
    )
    decompose_parser.add_argument(
        "--write-chunks", type=str, default=None, metavar="PATH",
        help="Write each streamed chunk to PATH as binary PKCP frames "
             "(symplectic x, z indices + complex coefficients). Writing "
             "runs only on the main drain thread - workers never touch "
             "the file. Without this flag (or --checkpoint-path), the "
             "CLI counts terms and discards chunk arrays. Requires "
             "--chunk-size on streamed/parallel paths. Same format as "
             "--checkpoint-path (resume-capable); prefer this name when "
             "you want a usable on-disk result. Read back with "
             "paulikit.algorithms.fwht.iter_checkpoint_chunks.",
    )
    decompose_parser.add_argument(
        "--checkpoint-path", type=str, default=None, metavar="PATH",
        help="Alias for --write-chunks: write each completed chunk to "
             "PATH so an interrupted run can resume with the same path. "
             "Omit for no on-disk writer (default).",
    )
    decompose_parser.add_argument(
        "--no-eager-threads", dest="eager_threads", action="store_false",
        help="With --parallel and --executor thread (or --executor "
             "auto resolving to thread), restore ThreadPoolExecutor's "
             "default LAZY thread spin-up instead of forcing all "
             "--n-workers OS threads to exist before the first chunk "
             "is submitted (the default). Measured directly: without "
             "eager spin-up, the first few chunks start staggered by "
             "~350-450us each rather than together, since each of the "
             "pool's first few submit() calls on a fresh pool forces "
             "a new OS thread creation. A one-time ~1.5-2ms cost "
             "either way - this only changes WHEN it is paid (all up "
             "front vs staggered across the first several chunks), "
             "not whether it is paid. No effect with --executor "
             "process or at --n-workers 1. Output is unaffected - "
             "this cannot change correctness.",
    )
    decompose_parser.set_defaults(func=cmd_decompose)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Time the decomposition across a sweep of N values",
        description=(
            "Runs the decompose workflow across multiple N values and "
            "prints a timing table."
        ),
    )
    benchmark_parser.add_argument(
        "--n-oscillators", "-n", type=_positive_int, nargs="+",
        default=[2, 4, 8, 16, 30],
        help="Space-separated list of N values to benchmark "
             "(default: 2 4 8 16 30)",
    )
    benchmark_parser.add_argument(
        "--atol", type=float, default=1e-10,
        help="Absolute coefficient threshold below which a term is "
             "dropped as zero (default: 1e-10)",
    )
    benchmark_parser.add_argument(
        "--compare-dense-sparse", action="store_true",
        help="Instead of the usual fwht_pauli_terms timing table, time "
             "fwht_pauli_coefficients's dense (sparse=False) and sparse "
             "(sparse=True) output modes directly, side by side, at "
             "each N (this is wall-clock timing only)",
    )
    benchmark_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Passed through to fwht_pauli_terms - see decompose "
             "--chunk-size's help. Ignored when --compare-dense-sparse is "
             "set (that path calls fwht_pauli_coefficients directly "
             "without chunking).",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    regenerate_parser = subparsers.add_parser(
        "regenerate-fixtures",
        help="Regenerate testing/fixtures.py's expected_terms constants (prints only)",
        description=(
            "Recomputes the N=2/N=4 expected Pauli decompositions using "
            "PennyLane as an independent oracle and prints them in a "
            "form ready to paste into paulikit/testing/fixtures.py. Does "
            "not modify that file automatically - see its module "
            "docstring for when you'd need to do this (only if "
            "paulikit.hamiltonian's Hamiltonian construction changes)."
        ),
    )
    regenerate_parser.set_defaults(func=cmd_regenerate_fixtures)

    return parser


def main(argv=None):
    """Entry point registered as the ``paulikit`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
