scNoah
======
   
Benchmarking cell type annotation and modality prediction.

.. py:currentmodule:: scparadise

Balance dataset
---------------
   
Balancing dataset using your own annotation for future model training.
Oversmaple or undersample some cell types.

.. automodule:: scnoah
   :members: balance, oversample, undersample

Annotation metrics
------------------

Test annotation method quality using confusion matrix, accuracy, balanced accuracy and calculating cell type specific precision, recall (also called sensitivity), specificity, f1-score, geometric mean, and index balanced accuracy of the geometric mean.

.. automodule:: scnoah
   :members: report_classif_full, report_classif_sens_spec, conf_matrix, pred_status

Regression metrics
------------------

Test modality prediction method quality using error metrics (RMSE, MedianAE, MeanAE), EVS, R² score and PC. 
Also, visualise metrics on cell embeddings.

RMSE - Root mean squared error
MeanAE - Mean absolute error
MedianAE - Median absolute error
EVS - Explained variance score
R² score - Coefficient of determination
PC - Pearson coefficient

For error metrics (RMSE, MedianAE, MeanAE): lower value - better prediction

.. automodule:: scnoah
   :members: report_reg, regres_status, pearson_coef_prot

Count cells
-----------

Count number of cell types per sample or condition.

.. automodule:: scnoah
   :members: cell_counter

