# scParadise
scParadise is a fast, tunable, high-throughput automatic cell type annotation and modality prediction framework. scParadise includes three sets of tools: scAdam - fast multi-task multi-class cell type annotation; scEve - modality prediction; scNoah - benchmarking cell type annotation and modality prediction. scParadise enables users to utilize a selection of pre-existing models (scAdam or scEve) as well as to develop and train custom models tailored to specific research needs. scNoah facilitates the evaluation of novel models and methods for automated cell type annotation and modality prediction in scRNA-seq analysis.

# Installation
## Using pip
```console
pip install scparadise
```

# Tutorials

[Using scAdam for cell type classification](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict.ipynb)

[Train custom scAdam model](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict.ipynb)

[Using scAdam in R](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict_R.R)

[Using scEve model to improve clusterization](https://github.com/Chechekhins/scParadise/blob/main/scripts_package/scAdam_predict.ipynb)

# Available models
### scAdam models
| Tissue/Model name | Description | Suspension | Accuracy | Balanced Accuracy | Number of Levels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PBMC  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.979 | 0.979 | 3 | 
| BMMC  | Bone marrow mononuclear cell of healthy adults  | cells | 0.947 | 0.942 | 3 | 
| Lung  | Core Human Lung Cell Atlas | cells | 0.965 | 0.964 | 5 | 
| Retina  | Single cell atlas of the human retina | cells | 0.984 | 0.979 | 4 | 

### scEve models
| Tissue/Model name | Description | Suspension | RMSE | MAE | Number of Levels |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PBMC  | Peripheral blood mononuclear cells of healthy adults 3' scRNA seq  | cells | 0.305 | 0.226 | 3 | 
| BMMC  | Bone marrow mononuclear cell of healthy adults  | cells | 0.706 | 0.454 | 3 | 

Mean AE - Mean Absolute Error

RMSE - Root Mean Squared Error

For error metrics (RMSE, Mean AE), a lower value indicates better prediction.

# Citation



 
