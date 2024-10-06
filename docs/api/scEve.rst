scEve
=====

Modality prediction using downloaded or custom scEve model.

.. py:currentmodule:: scparadise

Download model
--------------

Download pretrained model from `scParadise repository <https://github.com/Chechekhins/scParadise>`_.

.. automodule:: sceve
   :members: available_models, download_model

Model training and tuning
-------------------------

Train custom scEve model using your own multimodal dataset.
Hyperparameter tuning is also available to achieve better model performance.

.. automodule:: sceve
   :members: train, tune, train_tuned

Prediction 
----------

Modality prediction using downloaded or custom scEve model.

.. automodule:: sceve
   :members: predict
