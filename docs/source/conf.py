# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

# The project root is two levels up from this conf.py file (conf.py is in source/, which is in docs/)
# Example: if conf.py is /project/docs/source/conf.py, then project_root is /project
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root)) # Add project root to path
sys.path.insert(0, str(project_root / 'src')) # Add src directory to path

# This is crucial for autodoc to find your modules
# Ensure your Python project is installed in your environment or accessible via sys.path
# For example: pip install -e . (if you have setup.py or pyproject.toml)


# -- Project information -----------------------------------------------------

project = 'FinTech E-commerce Data Extractor'
copyright = '2025, Mikias Worku' # Replace with your name
author = 'Mikias Worku' # Replace with your name

# The full version, including alpha/beta/rc tags
release = '1.0.0' # Or match your project's version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',      # For automatic documentation from docstrings
    'sphinx.ext.napoleon',     # For Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode',     # Adds links to source code
    'sphinx.ext.intersphinx',  # For linking to other Sphinx docs (e.g., Python, NumPy)
    'sphinx.ext.todo',         # To use todo directives
    'myst_parser',             # To write docs in Markdown (.md files)
]

# Configure intersphinx for linking to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/', None),
    'datasets': ('https://huggingface.co/docs/datasets/', None),
    # Add other libraries your project uses that have Sphinx docs
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Choose your preferred theme: 'sphinx_rtd_theme' or 'furo'
html_theme = 'furo' # Or 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Set the URL base for the documentation when hosted on GitHub Pages
# This needs to match your GitHub Pages URL base path, typically /<repo-name>/
html_baseurl = '/amharic-ecommerce-data-extractor/' # REPLACE WITH YOUR REPO NAME (e.g., /my-repo-name/)

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False # Set to True if you use NumPy style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST Parser configuration (if using Markdown)
# Set the file suffixes for source files, including .md if using MyST-Parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# todo extension settings
todo_include_todos = True
