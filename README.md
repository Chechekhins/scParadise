# scParadise
scParadise is a fast, tunable, high-throughput automatic cell type annotation and modality prediction framework. scParadise includes three sets of tools: scAdam - fast multi-task multi-class cell type annotation; scEve - modality prediction; scNoah - benchmarking cell type annotation and modality prediction. scParadise enables users to utilize a selection of pre-existing models (scAdam or scEve) as well as to develop and train custom models tailored to specific research needs. scNoah facilitates the evaluation of novel models and methods for automated cell type annotation and modality prediction in scRNA-seq analysis.

![Graphical_abstract](https://github.com/user-attachments/assets/ccfc8fba-5eee-42c4-8486-3b5416bb4bd4)

# Installation
## Using pip
```console
pip install scparadise
```
## Create environment for using scparadise in R
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

## Create environment from scparadise.yml
1) Download [scparadise.yml](https://github.com/Chechekhins/scParadise/blob/main/scparadise.yml). 
2) Execute the following command in Anaconda (from directory with scparadise.yml):
```console
conda env create -f scparadise.yml
```
Installed environment includes scvi-tools, scanpy, muon, harmony, episcanpy, decoupler, scGen and other packages for scRNA-seq analysis.
Also environment includes jupyterlab and pytorch for GPU-accelerated model training.  

## GPU support
scParadise supports GPU usage automatically.

Required computing power: NVIDIA GeForce RTX 20 series or higher with installed driver.

# Tutorials

[Using scAdam for cell type classification](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict.ipynb)

[Train custom scAdam model](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scAdam/scAdam_train.ipynb)

[Using scAdam in R](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scAdam/R_scAdam_predict.R)

[Using scEve model to improve clusterization](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scEve/scEVE_clusterization.ipynb)

[Cross-species modality prediction using scEve](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scEve/Cross-species%20modality%20prediction%20using%20scEve.ipynb)

[Using scEve in R](https://github.com/Chechekhins/scParadise/blob/main/docs/tutorials/notebooks/scEve/scEve_predict_R.R)

# Tuning

scParadise supports both hyperparameter and fine tuning.
More information in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/)

# Available models
The full list of models is available in [scParadise documentation](https://scparadise.readthedocs.io/en/latest/models/index.html)
### scAdam models
| Tissue/Model name | Description | Suspension | Accuracy | Balanced Accuracy | Number of Levels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Human_PBMC  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.979 | 0.979 | 3 | 
| Human_BMMC  | Bone marrow mononuclear cell of healthy adults  | cells | 0.947 | 0.942 | 3 | 
| Human_Lung  | Core Human Lung Cell Atlas | cells | 0.965 | 0.964 | 5 | 
| Human_Retina  | Single cell atlas of the human retina | cells | 0.984 | 0.979 | 4 | 
| Mouse_Retina  | Single cell atlas of the mouse retina | cells | 0.967 | 0.960 | 4 | 
| Mouse_Cerebellum  | Single nuclei atlas of the Mouse cerebellum | nuclei | 0.999 | 0.999 | 2 | 
| Macaque_Cerebellum  | Single nuclei atlas of the Macaque cerebellum | nuclei | 0.995 | 0.994 | 2 | 

### scEve models
| Tissue/Model name | Description | Suspension | RMSE | MAE | Number of Proteins |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Human_PBMC_3p  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.305 | 0.226 | 224 | 
| Human_PBMC_5p  | Peripheral blood mononuclear cells of healthy adults 5' scRNA seq  | cells | 0.308 | 0.225 | 54 | 
| Human_BMMC  | Bone marrow mononuclear cell of healthy adults  | cells | 0.706 | 0.454 | 134 | 

Mean AE - Mean Absolute Error

RMSE - Root Mean Squared Error

For error metrics (RMSE, Mean AE), a lower value indicates better prediction.

# Contributing

We warmly welcome contributions to scParadise! If you have any ideas, enhancements, or bug fixes, please feel free to submit a pull request. Additionally, we encourage you to report any issues you encounter while using scParadise. Your feedback is invaluable in helping us improve the tool!

# Citation
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
 
