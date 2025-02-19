Installation
===================================

Using pip
---------

.. code-block:: python

   pip install scparadise -U


Create environment for using scparadise in R
--------------------------------------------

.. code-block:: bash

   conda create -n scparadise python=3.9

.. code-block:: bash

   conda activate scparadise

.. code-block:: python

   pip install scparadise -U

Set Python environment in R Studio: Tools - Global Options - Python

Create environment from scparadise.yml (recommended)
----------------------------------------------------

1. Download `scparadise.yml <https://github.com/Chechekhins/scParadise/blob/main/scparadise.yml>`_.
                             
2. Execute the following command in Anaconda (from directory with scparadise.yml):

.. code-block:: bash

   conda env create -f scparadise.yml

Installed environment includes latest scparadise, scvi-tools, scanpy, muon, harmony, jupyterlab and other packages for scRNA-seq analysis.

GPU support
-----------

scParadise supports GPU (NVIDIA) usage automatically. If you don't have GPU scParadise will automatically use CPU. 
We recommend using GPUs to train custom models as training on CPUs can take a long time.
