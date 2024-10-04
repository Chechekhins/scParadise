import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "scParadise"
repository_url = "https://github.com/Chechekhins/scParadise"
copyright = '2024, Vadim Chechekhin'
author = "Vadim Chechekhin"
release = "0.1.1b"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "nbsphinx_link",
    # "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_show_sourcelink = True
html_theme = "sphinx_book_theme"
html_title = project_name

html_logo = "_static/scParadise_logo.png"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "logo_only": True,
    "show_toc_level": 1,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "path_to_docs": "docs/",
    "repository_branch": version,
}
html_static_path = ["_static"]


# -- Options for nbshpinx ----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.0/configure.html

nbsphinx_execute = "never"
