scAdam
=======

Module for cell type annotation using downloaded or custom scAdam model.

.. module:: scparadise.scadam

.. currentmodule:: scparadise

Cross-tissue fast multi-level multi-class cell type annotation.

Download model
--------------
Download pretrained model from `scParadise repository <https://github.com/Chechekhins/scParadise>`_.

.. autosummary:: 
   :nosignatures:
   
   scadam.available_models
   scadam.download_model

Model training and tuning
-------------------------
Train custom scAdam model using your own reference dataset.
Hyperparameter tuning is also available to achieve better model performance.


.. autosummary::
   :nosignatures:
   
   scadam.train
   scadam.tune
   scadam.train_tuned

Prediction 
----------
Cell type annotation using downloaded or custom scAdam model.

.. autosummary:: 
   :nosignatures:
   
   scadam.predict
