scAdam
=======

Cell type annotation using downloaded or custom scAdam model.

```{autodoc2-summary}
.. currentmodule:: scparadise
```
Cross-tissue fast multi-level multi-class cell type annotation.

Download model
--------------
Download pretrained model from [scParadise repository](https://github.com/Chechekhins/scParadise).

```{autodoc2-summary}
.. autosummary::
   :toctree: generated
   :nosignatures:

   scadam.available_models
   scadam.download_model

```

Model training and tuning
-------------------------
Train custom scAdam model using your own reference dataset.
Hyperparameter tuning is also available to achieve better model performance.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   scadam.train
   scadam.tune
   scadam.train_tuned
```

Prediction 
----------
Cell type annotation using downloaded or custom scAdam model.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   scadam.predict
```
