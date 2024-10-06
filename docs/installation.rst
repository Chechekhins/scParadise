Installation
===================================

Using pip
---------

```{code-block} python
pip install scparadise
```

Create environment for using scparadise in R
--------------------------------------------

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

Create environment from scparadise.yml
--------------------------------------

1) Download [scparadise.yml](https://github.com/Chechekhins/scParadise/blob/main/scparadise.yml). 
                             
2) Execute the following command in Anaconda (from directory with scparadise.yml):
```console
conda env create -f scparadise.yml
```
Installed environment includes scvi-tools, scanpy, muon, harmony, jupyterlab and other packages for scRNA-seq analysis.

GPU support
-----------

scParadise supports GPU usage automatically. If you don't have GPU scParadise will automatically use CPU. 
We recommend using GPUs to train custom models as training on CPUs can take a long time.
