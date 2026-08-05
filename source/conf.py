# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------

project = 'qkrylov'
copyright = '2026, qkrylov contributors'
author = 'Subhajyoti Pal, Aritra Mukhopadhyay'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',    # For Markdown support
    'breathe',        # For Doxygen integration (C++)
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Breathe configuration ---------------------------------------------------

# breathe_projects = { "qkrylov": "../build/docs/doxygen/xml" }
# breathe_default_project = "qkrylov"
