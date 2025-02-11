.. image:: https://badge.fury.io/py/scparadise.svg?icon=si%3Apython
    :target: https://badge.fury.io/py/scparadise
.. image:: https://static.pepy.tech/badge/scparadise
    :target: https://pepy.tech/projects/scparadise
    :alt: PyPI Downloads
.. image:: https://readthedocs.org/projects/scparadise/badge/?version=latest
    :target: https://scparadise.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/scparadise?pypiBaseUrl=https%3A%2F%2Fpypi.org&logo=pypi&logoColor=blue
    :target: https://pypi.org/project/scparadise/
    :alt: PyPI - Version
.. image:: https://img.shields.io/badge/preprint-available-brightgreen
    :target: https://www.biorxiv.org/content/10.1101/2024.09.23.614509v1.full
    :alt: Preprint

Welcome to scParadise documentation!
===================================

`scParadise` is a fast, tunable, high-throughput automatic cell type annotation and modality prediction python framework.

`scParadise` includes three sets of tools: 

    1) `scAdam` - fast multi-task multi-class cell type annotation.
    2) `scEve` - modality prediction.
    3) `scNoah` - benchmarking cell type annotation and modality prediction. 

`scParadise` enables users to utilize a selection of pre-existing models (`scAdam` or `scEve`) 
as well as to develop, train and optimize custom models tailored to specific research needs. 

`scNoah` facilitates the evaluation of novel models and methods for automated cell type annotation 
and modality prediction in scRNA-seq analysis.

scParadise is now in active development. 

If you have any ideas, enhancements, or bug fixes, please feel free to submit a pull request in a `scParadise GitHub repo <https://github.com/Chechekhins/scParadise>`_.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Installation guide for scParadise.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Do you want your cells to be annotated and modalities predicted?

   .. grid-item-card:: Models
      :link: models/index
      :link-type: doc

      The table of scAdam and scEve models

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      The API reference of scParadise modules and functions


   .. grid-item-card:: Theory
      :link: theory
      :link-type: doc

      The theory of scParadise key concepts 

   .. grid-item-card:: GitHub
      :link: https://github.com/Chechekhins/scParadise

      The repository where you can try to find a solution of your problem


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   tutorials/index
   models/index
   api/index
   theory 
   GitHub <https://github.com/Chechekhins/scParadise>
   references
   citation

Small jokes from the scParadise team
************************************
So Adam gave names to all Monocytes, to the T cells of the blood, and to every living cell. But for Adam there was not found a helper comparable to him.

She became pregnant and gave birth to new modality. Eve said, “With the Lord's help, I have given birth to a surface proteins.”

Pairs of all functions that have the ability to check models quality in them came to Noah and entered the module. 
