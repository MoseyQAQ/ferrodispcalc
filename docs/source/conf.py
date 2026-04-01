import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ferrodispcalc'
copyright = '2025, Denan LI'
author = 'Denan LI'

from ferrodispcalc import __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'myst_parser',
              'sphinx.ext.napoleon',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',]

# Mock C++ extension and optional heavy dependencies so autodoc can import
# the package even when they are not installed in the doc-build environment.
autodoc_mock_imports = [
    'ferrodispcalc._cpp_bindings',
    'vispy',
    'PySide6',
    'OpenGL',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}