# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

"""GNU-style CLI help: usage layout, preserved newlines, examples."""

from paulikit.cli import build_parser
from paulikit.cli_help import HelpFormatter, gnu_usage


def test_gnu_usage_aligns_continuation_under_first_option():
    text = gnu_usage(
        "paulikit decompose",
        "[-h] [-n N]",
        "[--chunk-size CS]",
    )
    lines = text.splitlines()
    assert lines[0] == "%(prog)s [-h] [-n N]"
    # Continuations pad to len("usage: paulikit decompose ")
    assert lines[1].startswith(" " * len("usage: paulikit decompose "))
    assert "[--chunk-size CS]" in lines[1]


def test_help_formatter_keeps_author_newlines():
    fmt = HelpFormatter("prog")
    lines = fmt._split_lines("first line.\nsecond line with more words.", 40)
    assert lines[0].startswith("first line.")
    assert any(line.startswith("second line") for line in lines)


def test_fill_text_wraps_description_and_epilog_without_losing_breaks():
    """RawDescription keeps newlines but never wraps; we must do both."""
    fmt = HelpFormatter("paulikit")
    long_line = (
        "Exact Pauli decomposition of complex matrices. Peak memory "
        "can be bounded by chunk size rather than by the full term "
        "count when --stream or --parallel is used."
    )
    filled = fmt._fill_text(long_line, width=78, indent="")
    body_lines = [ln for ln in filled.splitlines() if ln]
    assert len(body_lines) >= 2
    assert all(len(ln) <= 78 for ln in body_lines)

    with_break = "First paragraph stays short.\n\nSecond paragraph also wraps."
    filled2 = fmt._fill_text(with_break, width=40, indent="")
    assert "\n\n" in filled2


def test_top_level_help_wraps_description_and_epilog():
    help_text = build_parser().format_help()
    # No single description/epilog line should blow past the formatter width.
    for line in help_text.splitlines():
        # Usage continuations are padded; still keep them reasonable.
        assert len(line) <= 90, repr(line)
    assert "Exact Pauli decomposition" in help_text
    assert "paulikit <command> --help" in help_text.replace("\n", " ")


def test_top_level_help_lists_commands_and_points_at_subcommand_help():
    help_text = build_parser().format_help()
    assert "decompose" in help_text
    assert "benchmark" in help_text
    assert "regenerate-fixtures" in help_text
    assert "paulikit <command> --help" in help_text.replace("\n", " ")


def test_decompose_help_has_examples_and_accurate_flags():
    parser = build_parser()
    decompose = None
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            if name == "decompose":
                decompose = sub
                break
    assert decompose is not None
    help_text = decompose.format_help()

    assert "Examples:" in help_text
    assert "paulikit decompose -n 4" in help_text
    assert "--write-chunks" in help_text
    assert "--progress" in help_text
    assert "PKCP" in help_text
    assert "fair warm-up" in help_text or "pre-count" in help_text
    # Usage must be the subcommand's own synopsis, not the parent usage
    # glued onto "decompose …".
    first = help_text.splitlines()[0]
    assert first.startswith("usage: paulikit decompose")
    assert "{decompose,benchmark" not in first
    # Continuations exist (GNU-style)
    assert any(
        line.startswith(" ") and "--chunk-size" in line
        for line in help_text.splitlines()[:8]
    )
    # Author newlines in description are kept (Examples on its own line)
    assert "\nExamples:\n" in help_text or "\nExamples:\r\n" in help_text



def test_benchmark_help_has_examples():
    parser = build_parser()
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            if name == "benchmark":
                help_text = sub.format_help()
                assert "Examples:" in help_text
                assert "paulikit benchmark" in help_text
                return
    raise AssertionError("benchmark subparser missing")


def test_nargs_plus_renders_as_metavar_ellipsis():
    parser = build_parser()
    for action in parser._subparsers._group_actions:
        for name, sub in action.choices.items():
            if name == "benchmark":
                help_text = sub.format_help()
                assert "N..." in help_text
                return
    raise AssertionError("benchmark subparser missing")
