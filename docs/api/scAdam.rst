scAdam
=======

Module for cell type annotation using downloaded or custom scAdam model.

.. :currentmodule:: scparadise

Cross-tissue fast multi-level multi-class cell type annotation.

Download model
--------------
Download pretrained model from [scParadise repository](https://github.com/Chechekhins/scParadise).

.. automodule:: scadam
   :members: 
   :private-members: available_models, download_model

Model training and tuning
-------------------------
Train custom scAdam model using your own reference dataset.
Hyperparameter tuning is also available to achieve better model performance.


.. automodule:: scadam
   :members: 
   :private-members: train, tune, train_tuned

Prediction 
----------
Cell type annotation using downloaded or custom scAdam model.

.. automodule:: scadam
   :members: 
   :private-members: predict
