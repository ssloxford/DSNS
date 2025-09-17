# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DSNS'
copyright = '2025, Joshua Smailes'
author = 'Joshua Smailes'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxarg.ext',
]

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    #'special-members': None,
    #'private-members': None,
    'undoc-members': True,
}

# Include __init__ methods in autodoc
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip
def setup(app):
    app.connect("autodoc-skip-member", skip)

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
