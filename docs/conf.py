# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
# Copyright (C) 2026 Beavernets Technologies

"""Sphinx configuration for paulikit's documentation.

Build with (from this directory, after `pip install -e ".[docs]"`
from the package root)::

    sphinx-build -b html . _build/html

or via the package Makefile::

    make docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paulikit  # noqa: E402

project = "paulikit"
copyright = "2026, Beavernets Technologies"
# Match CITATION.cff / pyproject.toml citation order.
author = (
    "Mohammadreza Khellat, Mohammad Masoumi, "
    "Saman Nasoori, Soroush Nasoori"
)
release = paulikit.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Agent plan/spec drafts belong under repo-root plans/ (gitignored),
    # never under docs/. Keep this pattern so a stray copy cannot leak
    # into the HTML build.
    "superpowers",
    "**/superpowers/**",
]

# README.md is written to read correctly as a plain file on
# Codeberg/GitHub (relative links like [verification/](verification/))
# and is included as-is into index.md via MyST's {include} directive
# rather than rewritten for Sphinx's cross-reference system. That
# directive naturally can't resolve those relative-link targets as
# Sphinx docs, so that specific, expected warning class is suppressed
# rather than distorting the README's plain-file readability to
# satisfy the docs build.
suppress_warnings = ["myst.xref_missing"]

# -- MyST (Markdown) configuration ------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    # theory.md, background.md and non_hermitian.md carry ~200 inline
    # LaTeX spans ($...$) and display blocks. Without dollarmath MyST
    # renders them as literal text, dollar signs and backslashes and
    # all, and emits no warning - so the build passes while the pages
    # are wrong.
    "dollarmath",
    "amsmath",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / Napoleon configuration ----------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
# Emits :ivar: fields instead of a generic "Attributes" list, which
# avoids duplicate-object warnings when autodoc also documents a
# dataclass's fields directly (see DecompositionFixture in
# paulikit.testing.fixtures).
napoleon_use_ivar = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output --------------------------------------------------------
html_theme = "furo"
html_title = f"paulikit {release}"
