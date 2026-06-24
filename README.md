# scParadise
[![PyPI version](https://badge.fury.io/py/scparadise.svg?icon=si%3Apython)](https://badge.fury.io/py/scparadise)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scparadise)
[![PyPI Downloads](https://static.pepy.tech/badge/scparadise)](https://pepy.tech/projects/scparadise)
[![Documentation](https://readthedocs.org/projects/scparadise/badge/?version=latest)](https://scparadise.readthedocs.io/en/latest/?badge=latest)
[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.09.23.614509v1.full)

Comprehensive information about the [installation](https://scparadise.readthedocs.io/en/latest/installation.html), [modules](https://scparadise.readthedocs.io/en/latest/theory.html), [tutorials](https://scparadise.readthedocs.io/en/latest/tutorials/index.html), and [API](https://scparadise.readthedocs.io/en/latest/api/index.html) of scParadise is available in the [scParadise documentation](https://scparadise.readthedocs.io/en/latest/index.html).

## Short overview
scParadise is an advanced Python framework designed for single-cell RNA sequencing (scRNA-seq) analysis, with a focus on accurate cell type annotation and modality prediction. It has three main tools:

1) scAdam specializes in multi-level cell type annotation and unknown cell type identification. Specifically, it identifies rare cell types that represent less than 1% of the population and uses deep learning techniques to improve the consistency and accuracy of diverse datasets. In addition, scAdam has the capability to identify cell types that were not used during model training. This enables scAdam to perform accurate cross-tissue cell type annotation and to detect novel, previously unknown cell types.
2) scEve specializes in imputation of new modalities. It facilitates the identification of specific cell subpopulations for targeting studies.
3) scNoah is a benchmarking tool used to evaluate the performance of various automatic cell type annotation and modality imputation methods.

scParadise enables users to utilize a selection of pre-existing models (scAdam or scEve) as well as to develop and train custom models tailored to specific research needs. scNoah facilitates the evaluation of novel models and methods for automated cell type annotation and modality imputation in scRNA-seq analysis.

![Graphical_abstract](https://github.com/user-attachments/assets/9a230527-2e08-4294-a7e7-3e39f48f7db1)

## Documentation
The full scParadise documentation, including [installation](https://scparadise.readthedocs.io/en/latest/installation.html), [usage](https://scparadise.readthedocs.io/en/latest/tutorials/index.html), and [API reference](https://scparadise.readthedocs.io/en/latest/api/index.html), is available at the [scParadise documentation](https://scparadise.readthedocs.io/en/latest/index.html)

## GPU support
scParadise supports GPU usage automatically.

Recommended computing power: NVIDIA GeForce RTX 20 series or higher with installed driver.

## Model optimization

scParadise supports both hyperparameter tuning and warm start training.
More information in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/tutorials/index.html)

## Available models
The full list of models is available in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/models/index.html)

Also full list of models is available using `scparadise.scadam.available_models()` for scAdam models and `scparadise.sceve.available_models()` for scEve models.

## Contributing

We warmly welcome contributions to scParadise! If you have any ideas, enhancements, or bug fixes, please feel free to submit a pull request. Additionally, we encourage you to report any issues you encounter while using scParadise. Your feedback is invaluable in helping us improve the tool!

## Citation
```bibtex
@article{10.1093/nar/gkag612,
    author = {Chechekhina, Elizaveta and Shcherbakova, Liya and Vigovskiy, Maksim and Tukhvatulin, Amir and Logunov, Denis and Tychinin, Dmitry and Svetlichnyy, Dmitry and Kulebyakin, Konstantin and Tkachuk, Vsevolod and Chechekhin, Vadim},
    title = {scParadise: tunable, highly accurate multi-level cell type annotation, unknown cell type identification, and modality imputation},
    journal = {Nucleic Acids Research},
    volume = {54},
    number = {11},
    pages = {gkag612},
    year = {2026},
    month = {06},
    issn = {1362-4962},
    doi = {10.1093/nar/gkag612},
    url = {https://doi.org/10.1093/nar/gkag612},
    eprint = {https://academic.oup.com/nar/article-pdf/54/11/gkag612/68547489/gkag612.pdf},
}
```
 
