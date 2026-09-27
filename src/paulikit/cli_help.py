# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
# Copyright (C) 2026 Beavernets Technologies
"""GNU-style ``--help`` formatting for the paulikit CLI.

Layout matches the measurements/Zenodo bench scripts: multi-line usage
aligned under the first option, author newlines preserved in help text,
and bold option names on a capable TTY. Kept inside the package so the
console script does not depend on the measurements tree.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_BOLD_ON = "\033[1m"
_BOLD_OFF = "\033[0m"


def ansi_enabled(stream=None) -> bool:
    """True when ANSI styling should be emitted on ``stream``."""
    if os.environ.get("NO_COLOR", ""):
        return False
    if os.environ.get("TERM", "") in ("", "dumb"):
        return False
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def bold(text: str, stream=None) -> str:
    """Wrap ``text`` in ANSI bold when allowed, else return unchanged."""
    if not text or not ansi_enabled(stream):
        return text
    return f"{_BOLD_ON}{text}{_BOLD_OFF}"


def _visible_len(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """GNU-friendly ``--help`` layout for paulikit.

    - ``nargs='+'`` renders as ``METAVAR...`` (not ``M [M ...]``).
    - Explicit newlines in ``help=`` / description / epilog are kept;
      each line is then wrapped to ``width`` (RawDescription alone
      leaves long single-line description/epilog unwrapped).
    - Option names are bold on a capable TTY (``NO_COLOR`` / non-TTY
      disables this), in the spirit of GNU coreutils ``--help``.
    """

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32, width=78)

    def _format_args(self, action, default_metavar):
        if action.nargs == argparse.ONE_OR_MORE:
            metavar = self._metavar_formatter(action, default_metavar)(1)[0]
            return f"{metavar}..."
        return super()._format_args(action, default_metavar)

    def _split_lines(self, text, width):
        lines: list[str] = []
        for paragraph in text.splitlines() or [""]:
            if not paragraph:
                lines.append("")
                continue
            stripped = paragraph.lstrip(" ")
            indent = paragraph[: len(paragraph) - len(stripped)]
            wrapped = textwrap.wrap(
                stripped,
                width=max(width - len(indent), 1),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            lines.extend(indent + part for part in wrapped)
        return lines

    def _fill_text(self, text, width, indent):
        # RawDescriptionHelpFormatter keeps author newlines but never
        # wraps — a single long description/epilog line stays one
        # terminal-wide blob. Wrap each author line; keep blanks and
        # leading indent (e.g. example command lines).
        # ``width`` already excludes ``indent`` (argparse convention
        # matches textwrap.fill(..., initial_indent=indent)).
        parts: list[str] = []
        for paragraph in text.splitlines() or [""]:
            if not paragraph:
                parts.append("")
                continue
            stripped = paragraph.lstrip(" ")
            lead = paragraph[: len(paragraph) - len(stripped)]
            avail = max(width - len(indent) - len(lead), 1)
            wrapped = textwrap.wrap(
                stripped,
                width=avail,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            parts.extend(f"{indent}{lead}{part}" for part in wrapped)
        return "\n".join(parts) + "\n"

    def _format_action(self, action):
        help_position = min(self._action_max_length + 2, self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        plain = super()._format_action_invocation(action)
        styled = bold(plain, stream=sys.stdout)

        if not action.help:
            action_header = f"{'':>{self._current_indent}}{styled}\n"
            parts = [action_header]
        elif _visible_len(plain) <= action_width:
            pad = action_width - _visible_len(plain)
            action_header = (
                f"{'':>{self._current_indent}}{styled}{' ' * pad}  "
            )
            indent_first = 0
            parts = [action_header]
        else:
            action_header = f"{'':>{self._current_indent}}{styled}\n"
            indent_first = help_position
            parts = [action_header]

        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append(f"{'':>{indent_first}}{help_lines[0]}\n")
                for line in help_lines[1:]:
                    parts.append(f"{'':>{help_position}}{line}\n")
        elif not action_header.endswith("\n"):
            parts.append("\n")

        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))
        return self._join_parts(parts)


def gnu_usage(prog: str, first_line: str, *more_lines: str) -> str:
    """Multi-line usage with continuations aligned under the first option."""
    pad = " " * len(f"usage: {prog} ")
    parts = [f"%(prog)s {first_line}"]
    parts.extend(f"{pad}{line}" for line in more_lines)
    return "\n".join(parts)


def arg_parser(**kwargs) -> argparse.ArgumentParser:
    """``ArgumentParser`` with paulikit's GNU help formatter."""
    import inspect

    kwargs.setdefault("formatter_class", HelpFormatter)
    supports_color = (
        "color" in inspect.signature(argparse.ArgumentParser.__init__).parameters
    )
    if supports_color:
        kwargs.setdefault("color", True)
    else:
        kwargs.pop("color", None)
    return argparse.ArgumentParser(**kwargs)
