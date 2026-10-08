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

import sys
import time

from paulikit import __version__
from paulikit.algorithms.fwht import (
    fwht_pauli_coefficients,
    fwht_pauli_terms,
    fwht_pauli_terms_iter,
)
from paulikit.cli_help import HelpFormatter, arg_parser, gnu_usage
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


class _ChunkProgress:
    """Opt-in main-thread progress on stderr for streamed CLI drains.

    Constructed only when ``--progress`` is on. The disabled path must
    not call into this class at all (separate drain loops) so the hot
    path stays free of per-chunk progress branches and method calls.

    ``n_total`` comes from the CLI's shared pre-drain
    ``count_stream_chunks`` call (also run when progress is off, so
    quiet and progress paths share the same CPU warm-up before the
    timed section).

    When enabled, most chunks pay only a cheap stride check
    (``n_chunks % stride``); ``time.perf_counter`` and I/O run on
    stride ticks that also pass a ~0.25s wall throttle, plus first and
    last. Library APIs never use this.
    """

    __slots__ = ("_n_total", "_tty", "_start", "_last_t", "_open_cr", "_stride")

    def __init__(self, n_total: int, *, stride: int = 32) -> None:
        self._n_total = max(0, int(n_total))
        self._tty = sys.stderr.isatty()
        self._start = time.perf_counter()
        self._last_t = 0.0
        self._open_cr = False
        self._stride = max(1, int(stride))

    def update(self, n_chunks: int, total_terms: int, *, force: bool = False) -> None:
        if not force and n_chunks > 1 and (n_chunks % self._stride) != 0:
            return
        now = time.perf_counter()
        if not force and n_chunks > 1 and (now - self._last_t) < 0.25:
            return
        self._last_t = now
        elapsed = now - self._start
        if self._n_total > 0:
            pct = 100.0 * n_chunks / self._n_total
            chunk_part = f"chunks={n_chunks}/{self._n_total} ({pct:.1f}%)"
            remaining = self._n_total - n_chunks
            if elapsed >= 0.5 and n_chunks >= 2 and remaining > 0:
                rate = n_chunks / elapsed
                eta_s = f"{remaining / rate:.1f}s"
            elif remaining <= 0:
                eta_s = "0.0s"
            else:
                eta_s = "--"
            line = (
                f"paulikit: {chunk_part}  terms={total_terms}  "
                f"elapsed={elapsed:.2f}s  eta={eta_s}"
            )
        else:
            line = (
                f"paulikit: chunks={n_chunks}  terms={total_terms}  "
                f"elapsed={elapsed:.2f}s"
            )
        if self._tty:
            print(f"\r{line}", end="", file=sys.stderr, flush=True)
            self._open_cr = True
        else:
            print(line, file=sys.stderr, flush=True)

    def finish(self, n_chunks: int, total_terms: int) -> None:
        self.update(n_chunks, total_terms, force=True)
        if self._open_cr:
            print(file=sys.stderr)
            self._open_cr = False


def _cmd_decompose_operator_file(args, write_path):
    """Dense out-of-core path: --operator-file + --parallel."""
    if not getattr(args, "parallel", False):
        print(
            "--operator-file requires --parallel "
            "(out-of-core dense uses parallel_decompose_arrays)",
            file=sys.stderr,
        )
        return 1
    if not args.chunk_size:
        print("--operator-file requires --chunk-size", file=sys.stderr)
        return 1
    if getattr(args, "stream", False):
        print(
            "--operator-file cannot be combined with --stream "
            "(use --parallel)",
            file=sys.stderr,
        )
        return 1

    from paulikit.algorithms.dense_bucketed import DenseBucketedSource
    from paulikit.algorithms.dense_input import resolve_dense_file
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    try:
        spec = resolve_dense_file(
            args.operator_file, meta=getattr(args, "operator_meta", None)
        )
    except (OSError, ValueError) as exc:
        print(f"--operator-file: {exc}", file=sys.stderr)
        return 1

    spill_dir = getattr(args, "spill_dir", None)
    print(
        f"Operator file {spec.path} ({spec.layout}), "
        f"dim={spec.dim} ({spec.n_qubits} qubits), "
        f"chunk_size={args.chunk_size}"
    )
    if spill_dir is not None:
        print(f"Spill directory: {spill_dir}")
    if write_path is not None:
        print(
            f"Writing streamed chunks to {write_path} "
            f"(symplectic x, z, coeff; resume-capable PKCP frames)"
        )

    try:
        source = DenseBucketedSource.from_dense_file(
            spec.path,
            meta=spec.meta_path,
            chunk_size=args.chunk_size,
            spill_dir=spill_dir,
            max_resident_buckets=getattr(args, "max_resident_buckets", 2),
        )
    except (OSError, ValueError) as exc:
        print(f"--operator-file: {exc}", file=sys.stderr)
        return 1

    n_chunks_planned = (spec.dim + args.chunk_size - 1) // args.chunk_size
    want_progress = bool(getattr(args, "progress", False))
    progress = _ChunkProgress(n_chunks_planned) if want_progress else None

    start = time.perf_counter()
    total_terms = 0
    n_chunks = 0
    stream = parallel_decompose_arrays(
        None,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        atol=args.atol,
        checkpoint_path=write_path,
        executor=args.executor,
        eager_threads=args.eager_threads,
        operator_source=source,
    )
    if progress is not None:
        for _x, _z, coeff in stream:
            n_chunks += 1
            total_terms += len(coeff)
            progress.update(n_chunks, total_terms)
        progress.finish(n_chunks, total_terms)
    else:
        for _x, _z, coeff in stream:
            n_chunks += 1
            total_terms += len(coeff)
    elapsed = time.perf_counter() - start

    print(
        f"Decomposition time (parallel file, executor={args.executor}): "
        f"{elapsed:.4f}s"
    )
    print(f"Chunks: {n_chunks}, nonzero Pauli terms: {total_terms}")
    if args.show_terms:
        print(
            "(--show-terms not available with --parallel: this path "
            "yields raw arrays and never builds labels; use "
            "terms_from_arrays on the chunks you actually need)"
        )
    return 0


def _cmd_decompose_operator_file(args, write_path):
    """Dense file-backed OOC path (layouts A/B via from_dense_file)."""
    if not getattr(args, "parallel", False):
        print(
            "--operator-file requires --parallel "
            "(out-of-core dense uses parallel_decompose_arrays)",
            file=sys.stderr,
        )
        return 1
    if not args.chunk_size:
        print("--operator-file requires --chunk-size", file=sys.stderr)
        return 1
    if getattr(args, "stream", False):
        print(
            "--operator-file cannot be combined with --stream "
            "(use --parallel)",
            file=sys.stderr,
        )
        return 1

    from paulikit.algorithms.dense_bucketed import DenseBucketedSource
    from paulikit.algorithms.dense_input import resolve_dense_file
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    try:
        spec = resolve_dense_file(
            args.operator_file, meta=getattr(args, "operator_meta", None)
        )
    except (OSError, ValueError) as exc:
        print(f"operator file error: {exc}", file=sys.stderr)
        return 1

    spill_dir = getattr(args, "spill_dir", None)
    print(
        f"operator-file={spec.path} layout={spec.layout} "
        f"dim={spec.dim} ({spec.n_qubits} qubits) codec={spec.codec}"
    )
    if write_path is not None:
        print(
            f"Writing streamed chunks to {write_path} "
            f"(symplectic x, z, coeff; resume-capable PKCP frames)"
        )

    try:
        source = DenseBucketedSource.from_dense_file(
            spec.path,
            meta=spec.meta_path,
            chunk_size=args.chunk_size,
            spill_dir=spill_dir,
            max_resident_buckets=getattr(args, "max_resident_buckets", 2),
        )
    except (OSError, ValueError) as exc:
        print(f"failed to build bucketed source: {exc}", file=sys.stderr)
        return 1

    n_chunks_planned = (spec.dim + args.chunk_size - 1) // args.chunk_size
    want_progress = bool(getattr(args, "progress", False))
    progress = _ChunkProgress(n_chunks_planned) if want_progress else None

    start = time.perf_counter()
    total_terms = 0
    n_chunks = 0
    stream = parallel_decompose_arrays(
        None,
        chunk_size=args.chunk_size,
        n_workers=args.n_workers,
        atol=args.atol,
        checkpoint_path=write_path,
        executor=args.executor,
        eager_threads=args.eager_threads,
        operator_source=source,
    )
    if progress is not None:
        for _x, _z, coeff in stream:
            n_chunks += 1
            total_terms += len(coeff)
            progress.update(n_chunks, total_terms)
        progress.finish(n_chunks, total_terms)
    else:
        for _x, _z, coeff in stream:
            n_chunks += 1
            total_terms += len(coeff)
    elapsed = time.perf_counter() - start

    print(
        f"Decomposition time (parallel file-backed, "
        f"executor={args.executor}): {elapsed:.4f}s"
    )
    print(f"Chunks: {n_chunks}, nonzero Pauli terms: {total_terms}")
    if args.show_terms:
        print(
            "(--show-terms not available with --parallel: this path "
            "yields raw arrays and never builds labels; use "
            "terms_from_arrays on the chunks you actually need)"
        )
    return 0


def cmd_decompose(args):
    """Build a synthetic N-oscillator Hamiltonian and Pauli-decompose it."""
    write_path, err = _resolve_chunk_write_path(args)
    if err:
        return err

    operator_file = getattr(args, "operator_file", None)
    if operator_file is not None:
        return _cmd_decompose_operator_file(args, write_path)

    n = args.n_oscillators
    spring_constants = _default_spring_constants(n)
    masses = _default_masses(n)

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

        from paulikit.algorithms.fwht import (
            count_stream_chunks,
            parallel_decompose_arrays,
        )

        # Always resolve the planned chunk count before the timed drain.
        # Same prep work the drain will redo - intentionally shared by
        # quiet and --progress paths so both start the timed section on
        # a warm CPU (DVFS), and --progress can show k/N without a
        # progress-only extra pass.
        n_chunks_planned = count_stream_chunks(padded, args.chunk_size)
        want_progress = bool(getattr(args, "progress", False))
        progress = (
            _ChunkProgress(n_chunks_planned) if want_progress else None
        )

        start = time.perf_counter()
        total_terms = 0
        n_chunks = 0
        stream = parallel_decompose_arrays(
            padded,
            chunk_size=args.chunk_size,
            n_workers=args.n_workers,
            atol=args.atol,
            checkpoint_path=write_path,
            executor=args.executor,
            eager_threads=args.eager_threads,
        )
        # Progress off: no per-chunk call/branch. Progress on: separate
        # loop body so the quiet path stays free of heartbeat cost.
        if progress is not None:
            for _x, _z, coeff in stream:
                n_chunks += 1
                total_terms += len(coeff)
                progress.update(n_chunks, total_terms)
            progress.finish(n_chunks, total_terms)
        else:
            for _x, _z, coeff in stream:
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

        from paulikit.algorithms.fwht import count_stream_chunks

        n_chunks_planned = count_stream_chunks(padded, args.chunk_size)
        want_progress = bool(getattr(args, "progress", False))
        progress = (
            _ChunkProgress(n_chunks_planned) if want_progress else None
        )

        start = time.perf_counter()
        total_terms = 0
        n_chunks = 0
        stream = fwht_pauli_terms_iter(
            padded,
            chunk_size=args.chunk_size,
            atol=args.atol,
            checkpoint_path=write_path,
            parallel_labels=args.parallel_labels,
        )
        if progress is not None:
            for chunk_terms in stream:
                n_chunks += 1
                total_terms += len(chunk_terms)
                progress.update(n_chunks, total_terms)
                if args.show_terms:
                    for label in sorted(chunk_terms):
                        print(f"  {label}: {chunk_terms[label]!r}")
            progress.finish(n_chunks, total_terms)
        else:
            for chunk_terms in stream:
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


_DECOMPOSE_HELP = """\
Demo/timing front end for the FWHT Pauli path: by default build a
synthetic coupled-oscillator Hamiltonian for N (fixed, deterministic
spring constants and masses — not a physical calibration), pad to a
power-of-two dimension, decompose it, and print wall time plus
nonzero term count.

With --operator-file, skip the synthetic Hamiltonian and stream a
dense on-disk operator (layout A: raw complex128 + sidecar JSON, or
layout B: square complex128 .npy) through the out-of-core bucketed
path. Requires --parallel and --chunk-size.

Default path (no --stream / --parallel) builds one label->coefficient
dict for the whole operator. Use --stream or --parallel when that
dict (or a dense matrix) would not fit in memory. Default CLI input
is dense; --parallel without --operator-file builds the Hamiltonian
sparse so large N stay reachable.

Examples:
  paulikit decompose -n 4
  paulikit decompose -n 4 --show-terms
  paulikit decompose -n 150 --parallel --chunk-size 2 --progress
  paulikit decompose -n 150 --parallel --chunk-size 2 \\
      --write-chunks /tmp/n150.pkcp
  paulikit decompose -n 50 --stream --chunk-size 4 --show-terms
  paulikit decompose --operator-file H.c128 --operator-meta H.c128.json \\
      --parallel --chunk-size 256 --spill-dir /tmp/H.buckets
  paulikit decompose --operator-file H.npy --parallel --chunk-size 256
"""

_BENCHMARK_HELP = """\
Run the same synthetic-Hamiltonian workflow as ``decompose`` across
several N values and print a timing table (N, qubits, wall time,
nonzero terms).

Examples:
  paulikit benchmark
  paulikit benchmark -n 16 30 50
  paulikit benchmark -n 4 8 --compare-dense-sparse
"""

_REGENERATE_HELP = """\
Recompute the N=2 / N=4 expected Pauli decompositions with PennyLane
as an independent oracle and print them ready to paste into
paulikit/testing/fixtures.py. Does not edit that file; see its module
docstring for when a hand update is needed (only if
paulikit.hamiltonian's construction changes).

Requires the development-only dependency PennyLane
(``pip install pennylane``).

Examples:
  paulikit regenerate-fixtures
"""


def build_parser():
    """Construct the top-level argparse.ArgumentParser for the paulikit CLI."""
    parser = arg_parser(
        prog="paulikit",
        usage=gnu_usage(
            "paulikit",
            "[-h] [--version]",
            "{decompose,benchmark,regenerate-fixtures} ...",
        ),
        description=(
            "Exact Pauli decomposition of complex matrices. Peak memory "
            "can be bounded by chunk size rather than by the full term "
            "count when --stream or --parallel is used."
        ),
        epilog=(
            "Run 'paulikit <command> --help' for that command's options "
            "and examples. See also the package README."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND",
        title="commands",
        # Without an explicit prog, argparse fills this from the parent
        # usage synopsis — which, once we set a multi-line GNU usage on
        # the parent, becomes an unreadable prefix on every subcommand.
        prog="paulikit",
    )

    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Demo: synthetic N-oscillator Hamiltonian, then Pauli-decompose it",
        formatter_class=HelpFormatter,
        usage=gnu_usage(
            "paulikit decompose",
            "[-h] [-n N | --operator-file PATH]",
            "[--operator-meta PATH] [--spill-dir DIR]",
            "[--atol ATOL] [--show-terms]",
            "[--sparse-output] [--chunk-size CS] [--stream]",
            "[--parallel-labels] [--parallel]",
            "[--executor {auto,thread,process}] [--n-workers N]",
            "[--write-chunks PATH | --checkpoint-path PATH]",
            "[--progress] [--no-eager-threads]",
        ),
        description=_DECOMPOSE_HELP,
    )
    decompose_parser.add_argument(
        "--n-oscillators", "-n", type=_positive_int, default=2, metavar="N",
        help="Number of coupled oscillators (default: 2). Larger N means "
             "more qubits after padding and, usually, far more Pauli "
             "terms. Ignored when --operator-file is set.",
    )
    decompose_parser.add_argument(
        "--operator-file", type=str, default=None, metavar="PATH",
        help="Dense on-disk operator instead of the synthetic Hamiltonian.\n"
             "Layout A: raw row-major complex128 + sidecar JSON "
             "(paulikit.dense_c128.v1).\n"
             "Layout B: square C-order complex128 .npy.\n"
             "Requires --parallel and --chunk-size. Streams via "
             "DenseBucketedSource (spill buckets; never loads full H).",
    )
    decompose_parser.add_argument(
        "--operator-meta", type=str, default=None, metavar="PATH",
        help="Sidecar JSON for --operator-file layout A "
             "(default: PATH.json next to the blob). Optional for .npy "
             "(may carry sha256).",
    )
    decompose_parser.add_argument(
        "--spill-dir", type=str, default=None, metavar="DIR",
        help="With --operator-file, directory for Pass-1 bucket spill "
             "files. Default: PATH.buckets next to the operator file. "
             "Prefer a local SSD (not network FS).",
    )
    decompose_parser.add_argument(
        "--max-resident-buckets", type=int, default=2, metavar="K",
        help="With --operator-file, LRU size for spilled buckets kept in "
             "RAM during gather (default: 2).",
    )
    decompose_parser.add_argument(
        "--atol", type=float, default=1e-10, metavar="ATOL",
        help="Drop a coefficient whose absolute value is strictly below "
             "this threshold (default: 1e-10). Same meaning as the "
             "library ``atol`` argument.",
    )
    decompose_parser.add_argument(
        "--show-terms", action="store_true",
        help="Print every nonzero Pauli string and its coefficient. Off "
             "by default because term counts grow quickly with N.\n"
             "With --stream, prints each chunk as it arrives.\n"
             "With --parallel, labels are never built: the flag prints a "
             "short notice and the term count only.",
    )
    decompose_parser.add_argument(
        "--sparse-output", action="store_true",
        help="Call fwht_pauli_coefficients(sparse=True) and report its "
             "timing and active-row count, instead of the usual "
             "label->coefficient dict from fwht_pauli_terms.\n"
             "fwht_pauli_terms already uses the sparse path internally; "
             "this flag is for inspecting the raw sparse array shape "
             "itself. Only used on the default (non-stream, "
             "non-parallel) path — --stream / --parallel take "
             "precedence if also set.",
    )
    decompose_parser.add_argument(
        "--chunk-size", type=int, default=None, metavar="CS",
        help="Process active rows in blocks of at most CS instead of one "
             "(n_active, dim) array, bounding peak memory to roughly "
             "CS * dim complex entries.\n"
             "Required with --stream and with --parallel.\n"
             "On the library APIs (parallel_decompose / "
             "parallel_decompose_arrays), omitting chunk_size auto-tunes "
             "against measured cache boundaries; this CLI flag is always "
             "explicit when set.",
    )
    decompose_parser.add_argument(
        "--stream", action="store_true",
        help="Use fwht_pauli_terms_iter: yield one label->coefficient "
             "dict per chunk instead of one combined dict for the whole "
             "operator, so peak memory does not grow with total term "
             "count.\n"
             "Needed when the full result would not fit (e.g. tens of "
             "millions of terms at large N). Requires --chunk-size.",
    )
    decompose_parser.add_argument(
        "--parallel-labels", action="store_true",
        help="With --stream, use the oneTBB-parallel label kernel for "
             "each chunk instead of the serial one.\n"
             "Wins about 1.1-1.4x wall-clock in isolation, but usually "
             "adds little once embedded in the streaming pipeline at "
             "large N (dict construction dominates labeling there).\n"
             "Ignored without --stream.",
    )
    decompose_parser.add_argument(
        "--parallel", action="store_true",
        help="Decompose across workers via parallel_decompose_arrays, "
             "streaming raw (x, z, coeff) arrays per chunk rather than "
             "building Pauli labels.\n"
             "This is the path that scales: large-N runs finish in about "
             "a second on a typical workstation while holding tens of "
             "MiB, and it reaches sizes a dense matrix cannot hold "
             "(15 qubits ~16 GiB dense; 16 qubits ~64 GiB).\n"
             "Requires --chunk-size. Builds the Hamiltonian sparse. "
             "--show-terms cannot print labels on this path.",
    )
    decompose_parser.add_argument(
        "--executor", choices=("auto", "thread", "process"), default="auto",
        metavar="{auto,thread,process}",
        help="With --parallel, how to run chunk work (default: auto).\n"
             "  thread   — concurrent compiled kernels, no pickling "
             "(both release the GIL).\n"
             "  process  — process pool; pays IPC but does not need "
             "compiled kernels.\n"
             "  auto     — thread when compiled kernels are available, "
             "else process.\n"
             "Wrong choice can cost ~6x either way, so auto decides per "
             "install rather than globally.",
    )
    decompose_parser.add_argument(
        "--n-workers", type=int, default=None, metavar="N",
        help="With --parallel, worker count. Default: number of distinct "
             "physical cores (not logical CPUs — hyperthread siblings "
             "share execution units and measured worse).\n"
             "On a 4-core machine, 4 threads reached ~3.44x vs an Amdahl "
             "ceiling of ~3.63x; 8 threads spent more cycles for no "
             "wall-clock gain.",
    )
    decompose_parser.add_argument(
        "--write-chunks", type=str, default=None, metavar="PATH",
        help="Write each streamed chunk to PATH as binary PKCP frames "
             "(symplectic x, z indices + complex coefficients).\n"
             "I/O runs only on the main drain thread — workers never "
             "touch the file. Without this flag (or --checkpoint-path), "
             "the CLI counts terms and discards chunk arrays.\n"
             "Requires --chunk-size on --stream / --parallel. Same "
             "resume-capable format as --checkpoint-path; prefer this "
             "name when you want a usable on-disk result.\n"
             "Read back with "
             "paulikit.algorithms.fwht.iter_checkpoint_chunks.",
    )
    decompose_parser.add_argument(
        "--checkpoint-path", type=str, default=None, metavar="PATH",
        help="Alias for --write-chunks: write each completed chunk to "
             "PATH so an interrupted run can resume with the same path.\n"
             "Omit for no on-disk writer (default).",
    )
    decompose_parser.add_argument(
        "--progress", action="store_true",
        help="With --parallel or --stream, print chunk progress on "
             "stderr from the main drain thread only (k/N, percent, "
             "ETA; stride-throttled; TTY overwrites one line).\n"
             "Off by default: the quiet drain loop has no per-chunk "
             "progress calls, so scripts, CI, and measurement harnesses "
             "stay quiet. Both quiet and progress paths still pre-count "
             "planned chunks once before the timed drain (fair warm-up).\n"
             "Library APIs never emit progress.",
    )
    decompose_parser.add_argument(
        "--no-eager-threads", dest="eager_threads", action="store_false",
        help="With --parallel and a thread executor (explicit or auto), "
             "keep ThreadPoolExecutor's lazy thread spin-up instead of "
             "creating all --n-workers OS threads before the first "
             "submit (the default).\n"
             "Without eager spin-up, the first few chunks start "
             "staggered by ~350-450 us each while threads are created. "
             "The one-time cost (~1.5-2 ms) is paid either way — this "
             "only changes when.\n"
             "No effect with --executor process or --n-workers 1. "
             "Does not change numerical results.",
    )
    decompose_parser.set_defaults(func=cmd_decompose)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Time the decomposition across a sweep of N values",
        formatter_class=HelpFormatter,
        usage=gnu_usage(
            "paulikit benchmark",
            "[-h] [-n N...] [--atol ATOL]",
            "[--compare-dense-sparse] [--chunk-size CS]",
        ),
        description=_BENCHMARK_HELP,
    )
    benchmark_parser.add_argument(
        "--n-oscillators", "-n", type=_positive_int, nargs="+",
        default=[2, 4, 8, 16, 30], metavar="N",
        help="One or more oscillator counts to time "
             "(default: 2 4 8 16 30).",
    )
    benchmark_parser.add_argument(
        "--atol", type=float, default=1e-10, metavar="ATOL",
        help="Absolute coefficient threshold; terms below this are "
             "dropped (default: 1e-10).",
    )
    benchmark_parser.add_argument(
        "--compare-dense-sparse", action="store_true",
        help="Instead of the usual fwht_pauli_terms table, time "
             "fwht_pauli_coefficients dense (sparse=False) vs sparse "
             "(sparse=True) side by side at each N.\n"
             "Wall-clock only; no correctness check.",
    )
    benchmark_parser.add_argument(
        "--chunk-size", type=int, default=None, metavar="CS",
        help="Passed through to fwht_pauli_terms (see "
             "``paulikit decompose --chunk-size``).\n"
             "Ignored with --compare-dense-sparse (that path calls "
             "fwht_pauli_coefficients without chunking).",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    regenerate_parser = subparsers.add_parser(
        "regenerate-fixtures",
        help=(
            "Print regenerated FIXTURE_N2/N4 constants "
            "(PennyLane; does not edit files)"
        ),
        formatter_class=HelpFormatter,
        usage=gnu_usage("paulikit regenerate-fixtures", "[-h]"),
        description=_REGENERATE_HELP,
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
