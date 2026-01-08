# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "socca"
copyright = "2025, Luca Di Mascolo"
author = "Luca Di Mascolo"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

ISSUE_URL = (
    "https://github.com/lucadimascolo/socca/issues/new"
    + "?assignees=&labels=&projects=&template={}.md&title="
)

# html_theme = "sphinxawesome_theme"
# html_theme = "breeze"
# html_theme = "sphinx_book_theme"
# html_theme = "shibuya"
html_theme = "furo"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
]

# Napoleon settings to make attributes explicit
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_ivar = True  # Use ":ivar:" for instance attributes
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def run_citation_generator(app):
    script = Path(__file__).parent / "_scripts" / "generate_citation.py"
    subprocess.check_call(["python", str(script)])


def setup(app):
    app.connect("builder-inited", run_citation_generator)
