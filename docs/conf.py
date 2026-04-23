"""Sphinx configuration for pypfda."""

from __future__ import annotations

from importlib.metadata import version as pkg_version

project = "pypfda"
author = "Bijan Fallah"
copyright = "2026, Bijan Fallah"

try:
    release = pkg_version("pypfda")
except Exception:  # pragma: no cover
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "tasklist",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"pypfda {version}"
html_theme_options = {
    "source_repository": "https://github.com/bijanf/pypfda/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bijanf/pypfda",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

# Treat warnings as errors in CI; pyproject sets fail_on_warning for RTD.
nitpicky = False

# Suppress autosummary-related duplicate-object warnings caused by
# `autosummary` + `autodoc_typehints` documenting dataclass attributes
# from both the package re-export and the underlying submodule. The
# resulting docs are correct; the warning is cosmetic.
suppress_warnings = ["autosectionlabel.*"]
