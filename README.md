# scParadise
scParadise is a fast, tunable, high-throughput automatic cell type annotation and modality prediction framework. scParadise includes three sets of tools: scAdam - fast multi-task multi-class cell type annotation; scEve - modality prediction; scNoah - benchmarking cell type annotation and modality prediction. scParadise enables users to utilize a selection of pre-existing models (scAdam or scEve) as well as to develop and train custom models tailored to specific research needs. scNoah facilitates the evaluation of novel models and methods for automated cell type annotation and modality prediction in scRNA-seq analysis.

![Graphical abstract.tif](https://github.com/Chechekhins/scParadise/blob/main/Graphical%20abstract.tif)

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
1) Download scparadise.yml. 
2) Execute the following command in Anaconda (from directory with scparadise.yml):
```console
conda env create -f scparadise.yml
```
Installed environment includes scvi-tools, scanpy, muon, harmony, jupyterlab and other packages for scRNA-seq analysis.

## GPU support
scParadise supports GPU usage automatically.

Required computing power: NVIDIA GeForce RTX 20 series or higher with installed driver.

# Tutorials

[Using scAdam for cell type classification](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict.ipynb)

[Train custom scAdam model](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_train.ipynb)

[Using scAdam in R](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict_R.R)

[Using scEve model to improve clusterization](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scEVE_clusterization.ipynb)

[Using scEve in R](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scEve_predict_R.R)

# Available models
### scAdam models
| Tissue/Model name | Description | Suspension | Accuracy | Balanced Accuracy | Number of Levels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Human_PBMC  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.979 | 0.979 | 3 | 
| Human_BMMC  | Bone marrow mononuclear cell of healthy adults  | cells | 0.947 | 0.942 | 3 | 
| Human_Lung  | Core Human Lung Cell Atlas | cells | 0.965 | 0.964 | 5 | 
| Human_Retina  | Single cell atlas of the human retina | cells | 0.984 | 0.979 | 4 | 
| Mouse_Retina  | Single cell atlas of the mouse retina | cells | 0.967 | 0.960 | 4 | 

### scEve models
| Tissue/Model name | Description | Suspension | RMSE | MAE | Number of Proteins |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Human_PBMC  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.305 | 0.226 | 224 | 
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


 
