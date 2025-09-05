# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add the path to your package if needed
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
project = 'PyPL'
author = 'Yu Jin'
copyright = f'{datetime.now().year}, {author}'
release = '0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # Supports NumPy and Google docstrings
    'sphinx.ext.viewcode',      # Adds links to source code
    'sphinx.ext.mathjax',       # Renders LaTeX math
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Use Google or NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc options ---------------------------------------------------------
autoclass_content = 'class'     # Include class docstring only
autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = True

# -- Other options -----------------------------------------------------------
# Set master doc
master_doc = 'index'

# Fix for mathjax rendering on RTD-style themes
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

