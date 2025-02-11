# scParadise
[![PyPI Downloads](https://static.pepy.tech/badge/scparadise)](https://pepy.tech/projects/scparadise)
[![Documentation](https://readthedocs.org/projects/scparadise/badge/?version=latest)](https://scparadise.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/scparadise.svg?icon=si%3Apython)](https://badge.fury.io/py/scparadise)

Comprehensive information about the [installation](https://scparadise.readthedocs.io/en/latest/installation.html), [modules](https://scparadise.readthedocs.io/en/latest/theory.html), [tutorials](https://scparadise.readthedocs.io/en/latest/tutorials/index.html), and [API](https://scparadise.readthedocs.io/en/latest/api/index.html) of scParadise is available in the [scParadise documentation](https://scparadise.readthedocs.io/en/latest/index.html).

## Short overview
scParadise is an advanced Python framework designed for single-cell RNA sequencing (scRNA-seq) analysis, with a focus on accurate cell type annotation and modality prediction. It has three main tools:

1) scAdam specializes in multilevel cell type annotation. Specifically, it identifies rare cell types that represent less than 1% of the population and uses deep learning techniques to improve the consistency and accuracy of diverse datasets.
2) scEve aims to improve the prediction of surface protein markers. Clustering and separating cell types It facilitates the identification of specific cell subpopulations for targeting studies.
3) scNoah is a benchmarking tool used to evaluate the performance of various automatic cell type annotation and modality prediction methods.

scParadise enables users to utilize a selection of pre-existing models (scAdam or scEve) as well as to develop and train custom models tailored to specific research needs. scNoah facilitates the evaluation of novel models and methods for automated cell type annotation and modality prediction in scRNA-seq analysis.

![Graphical_abstract](https://github.com/user-attachments/assets/ccfc8fba-5eee-42c4-8486-3b5416bb4bd4)

## Installation
### Using pip
```console
pip install scparadise
```
### Create environment for using scparadise in R
```console
conda create -n scparadise python=3.9
```
```console
conda activate scparadise
```
```console
pip install scparadise
```
Set Python environment in R Studio: Tools - Global Options - Python

### Create environment from scparadise.yml
1) Download [scparadise.yml](https://github.com/Chechekhins/scParadise/blob/main/scparadise.yml). 
2) Execute the following command in Anaconda (from directory with scparadise.yml):
```console
conda env create -f scparadise.yml
```
Installed environment includes scvi-tools, scanpy, muon, harmony, episcanpy, decoupler, scGen and other packages for scRNA-seq analysis.
Also environment includes jupyterlab and pytorch for GPU-accelerated model training.  

## GPU support
scParadise supports GPU usage automatically.

Recommended computing power: NVIDIA GeForce RTX 20 series or higher with installed driver.

## Tutorials

[Using scAdam for cell type classification](https://scparadise.readthedocs.io/en/latest/tutorials/notebooks/scAdam/scAdam_predict.html)

[Train custom scAdam model](https://scparadise.readthedocs.io/en/latest/tutorials/notebooks/scAdam/scAdam_train.html)

[Using scAdam in R](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scAdam/R_scAdam_predict.R)

[Using scEve model to improve clusterization](https://scparadise.readthedocs.io/en/latest/tutorials/notebooks/scEve/scEve_clusterization.html)

[Cross-species modality prediction using scEve](https://scparadise.readthedocs.io/en/latest/tutorials/notebooks/scEve/Cross_species_modality_prediction_using_scEve.html)

[Using scEve in R](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scEve/scEve_predict_R.R)

## Model optimization

scParadise supports both hyperparameter tuning and warm start training.
More information in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/tutorials/notebooks/scAdam/scAdam_model_optimization.html)

## Available models
The full list of models is available in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/models/index.html)

Also full list of models is available using `scparadise.scadam.available_models()` for scAdam models and `scparadise.sceve.available_models()` for scEve models.

## Contributing

We warmly welcome contributions to scParadise! If you have any ideas, enhancements, or bug fixes, please feel free to submit a pull request. Additionally, we encourage you to report any issues you encounter while using scParadise. Your feedback is invaluable in helping us improve the tool!

## Citation
```bibtex
@article {Chechekhina2024.09.23.614509,
	author = {Chechekhina, Elizaveta and Tkachuk, Vsevolod and Chechekhin, Vadim},
	title = {scParadise: Tunable highly accurate multi-task cell type annotation and surface protein abundance prediction},
	year = {2024},
	doi = {10.1101/2024.09.23.614509},
	URL = {https://www.biorxiv.org/content/early/2024/09/24/2024.09.23.614509},
	journal = {bioRxiv}
 }
```
 
