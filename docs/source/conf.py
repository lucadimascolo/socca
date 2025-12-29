# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'socca'
copyright = '2025, Luca Di Mascolo'
author = 'Luca Di Mascolo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

ISSUE_URL = ("https://github.com/lucadimascolo/socca/issues/new"
          + "?assignees=&labels=&projects=&template={}.md&title="
)

#html_theme = "sphinxawesome_theme"
#html_theme = "breeze"
#html_theme = "sphinx_book_theme"
#html_theme = "shibuya"
html_theme = "furo"

html_static_path = ['_static']
html_css_files = ['custom.css']

# html_theme_options = {
#   "accent_color": "tomato",
# }

'''
html_theme_options = {
    "nav_links_align": "right",
    "nav_links": [
        {
            "title": "Wanna help?",
            "url": "wannahelp",
            "children": [
                {
                    "title": "Report an issue",
                    "url": ISSUE_URL.format("bug_report"),
                    "summary": "Found a bug? Let us know!",
                    "external": True,
                },
                {
                    "title": "Request a feature",
                    "url": ISSUE_URL.format("feature_request"),
                    "summary": "Would you like to see a new feature?",
                    "external": True,
                },
                {
                    "title": "Contribute",
                    "url": "https://github.com/lucadimascolo/socca/pulls",
                    "summary": "Submit a pull request",
                    "external": True,
                },

            ],
        },
    ]
}
'''

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

import subprocess
from pathlib import Path

def run_citation_generator(app):
    script = Path(__file__).parent / "_scripts" / "generate_citation.py"
    subprocess.check_call(["python", str(script)])

def setup(app):
    app.connect("builder-inited", run_citation_generator)