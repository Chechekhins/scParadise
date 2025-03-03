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
   :toctree: generated/

   scadam.available_models
   scadam.download_model

Model training and optimization
-------------------------
Train custom scAdam model using your own reference dataset.
Warm start training and hyperparameter tuning are also available to achieve better model performance.


.. autosummary::
   :nosignatures:
   :toctree: generated/
   
   scadam.train
   scadam.warm_start
   scadam.hyperparameter_tuning
   scadam.train_tuned

Prediction 
----------
Cell type annotation using downloaded or custom scAdam model.

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   
   scadam.predict
