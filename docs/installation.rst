Installation
===================================

Using pip
---------

.. code-block:: python

   pip install scparadise -U


Create environment for using scparadise
--------------------------------------------

.. code-block:: bash

   conda create -n scparadise python=3.10

.. code-block:: bash

   conda activate scparadise

.. code-block:: python

   pip install scparadise -U

If you want to use scParadise from R, you need to configure a Python environment in RStudio: Tools - Global Options - Python

Create environment from scparadise.yml (recommended)
----------------------------------------------------

1. Download `scparadise_3.10.yaml <https://github.com/Chechekhins/scParadise/blob/main/scparadise_3.10.yaml>`_ or `scparadise_3.11.yaml <https://github.com/Chechekhins/scParadise/blob/main/scparadise_3.11.yaml>`_.

2. Install g++ (optional, for a clean installation):

.. code-block:: bash

   sudo apt update
   sudo apt-get install g++
                             
3. Execute the following command in Anaconda (from directory with scparadise_3.10.yaml or scparadise_3.11.yaml):

.. code-block:: bash

   conda env create -f scparadise_3.10.yaml

The installed environment is based on Python 3.10 (scparadise_3.10) or 3.11 (scparadise_3.11) and includes the latest version of scparadise, scvi-tools, scanpy, muon, harmony, jupyterlab, liana, decoupler and other packages for scRNA-seq analysis.

GPU support
-----------

scParadise automatically uses an NVIDIA GPU if available. If you do not have a GPU, scParadise will fall back to the CPU. 
We recommend using GPUs to train custom models, as training on CPUs can take a long time.
