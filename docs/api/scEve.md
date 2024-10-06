# scEve

Modality prediction using downloaded or custom scEve model.

```{eval-rst}
.. currentmodule:: scparadise
```

## Download model

Download pretrained model from [scParadise repository](https://github.com/Chechekhins/scParadise).

```{eval-rst}
.. autosummary::
   :nosignatures:

   sceve.available_models
   sceve.download_model

```

## Model training and tuning

Train custom scEve model using your own multimodal dataset.
Hyperparameter tuning is also available to achieve better model performance.

```{eval-rst}
.. autosummary::
   :nosignatures:

   sceve.train
   sceve.tune
   sceve.train_tuned
```

## Prediction 

Modality prediction using downloaded or custom scEve model.

```{eval-rst}
.. autosummary::
   :nosignatures:

   sceve.predict
```
