Theory
######
This section aims to clarify the key concepts that underpin the operation of the scParadise library. Below are the characteristics of the scAdam and scEve models. Additionally, the quality control metrics used by scNoah are explained.

Model
*****
A machine learning model is a computational framework or program that has been trained to recognize patterns and make predictions based on input data. It is the result of applying a machine learning algorithm to a dataset, allowing the model to learn from the data and generalize its knowledge to new, unseen instances.

Key Characteristics
===================

1. Training: Machine learning models are created through a training process where they learn from data. During this phase, the algorithm optimizes its parameters to minimize prediction errors, resulting in a model capable of making accurate predictions.

2. Types of Learning:

   * Supervised Learning: The model is trained on labeled data, where each input is paired with the correct output (e.g., classification tasks).
  
   *	Unsupervised Learning: The model learns from unlabeled data to identify patterns or groupings without explicit instructions (e.g., clustering).
   *	Semi-Supervised Learning: Combines labeled and unlabeled data for training, enhancing performance when labeled data is limited.
   *	Reinforcement Learning: The model learns by interacting with an environment and receiving feedback in the form of rewards or penalties.

3. Output: Once trained, a machine learning model can make predictions on new data. For example, it can classify images, predict numerical values, or recognize speech based on previously unseen inputs.

4. Storage: Machine learning models can be saved as files or objects, allowing them to be reused for future predictions without needing retraining.

scAdam models
*************
The scAdam model is a component of the scParadise framework, which is designed for fast reference-free multi-level multi-label cell type annotation. 

Overview of scAdam
==================
*	Purpose: scAdam is primarily used for multi-level cell type annotation in single-cell datasets. It aims to enhance the accuracy and consistency of cell type predictions, particularly for rare cell types that are often challenging to identify with traditional methods.
*	Functionality: The model incorporates several key steps in its pipeline:
    1.	Feature Selection (optional): scAdam begins by selecting significant features (genes) that are most relevant for the classification task. This includes identifying highly variable genes and excluding those that do not contribute meaningfully to distinguishing between different cell types.
    2.	Automated Dataset Balancing (optional): Given the imbalanced nature of many single-cell datasets (where certain cell types are underrepresented), scAdam employs techniques to balance the dataset, ensuring that all classes are adequately represented during training.
    3.	Model Training: The model is then trained using the selected features and balanced data, allowing it to learn patterns associated with different cell types. We recommend using **balanced_accuracy** as an evaluation metric.

Key Features
============

*	High Accuracy and Balanced Accuracy: scAdam has been shown to surpass existing methods in annotating rare cell types, achieving high average accuracy and balanced_accuracy across diverse datasets.

*	Robustness: The model provides consistent results even when applied to different test datasets, which is crucial for reproducibility in scientific research.

*	Multi-task Learning: scAdam supports multitasking capabilities, enabling it to extract individual cell types for more targeted investigations.

Applications
============

scAdam is particularly valuable in biomedical research where understanding cellular composition and interactions within tissues is critical. By providing accurate annotations of cell types from complex single-cell datasets, it aids researchers in exploring tissue architecture and cellular functions more effectively. 

scEve models
*************


scNoah metrics
**************

The scNoah models are part of the scParadise framework, which is designed for benchmarking of cell type annotation methods and modality prediction in scRNA-seq data.

Overview of scNoah
==================
*	Purpose: scNoah serves as a benchmarking tool within the scParadise framework. Its primary function is to evaluate the performance of cell type annotation and modality prediction methods, ensuring that these processes are reliable and accurate.

*	Functionality:
   1. Unified Benchmarking: scNoah provides a unified approach to assess various automatic cell type annotation methods and modality prediction techniques. This is crucial for comparing different models and understanding their strengths and weaknesses.
   2. Comprehensive Metrics: The model employs a range of classic machine learning metrics, such as accuracy, balanced accuracy, precision, sensitivity, specificity, F1-score, and geometric mean. These metrics help in evaluating the quality of predictions made by different models.
Key Features
============

*	Visualization Tools: scNoah includes tools for visualizing prediction performance using normalized confusion matrices. This allows researchers to see how well each model performs across different cell types, highlighting areas where predictions may be inconsistent or inaccurate.

*	Detailed Quality Assessment: The model emphasizes the need for a thorough evaluation of cell type annotation methods by recommending the use of multiple test datasets. This approach helps ensure reproducibility and reliability in predictions across diverse datasets.

*	Support for Modality Prediction: In addition to benchmarking cell type annotation, scNoah also facilitates the assessment of modality prediction methods, making it a versatile tool within the scParadise framework.

Applications
============
scNoah is particularly useful in:

*	Comparative Studies: scNoah can be used to evaluate and compare the effectiveness of various existing methods for cell type annotation and modality prediction, assisting in the selection of the most appropriate approach for specific datasets.

*	Quality Control: By providing detailed metrics and visualizations, scNoah helps maintain high standards in the analysis of single-cell data, ensuring that findings are robust and reproducible. 

.. figure:: 
   :align: center
   _static/TP_TN_FP_FN.png

Where:
1. True Positives (TP): The number of correct positive predictions made by the model.
2. False Positives (FP): The number of incorrect positive predictions made by the model.
3. True Negatives (TN): The number of correct negative predictions made by the model (model accurately identified instances that do not belong to the positive class).
4. False Negatives (FN): The number of actual positive instances that were incorrectly predicted as negative by the model.

Precision
*********
Precision is a key metric in machine learning that evaluates the accuracy of a model's positive predictions. It is defined as the ratio of true positive predictions to the total number of instances predicted as positive (which includes both true positives and false positives). Usefull for scAdam model quality control.

Mathematically, precision can be expressed as:

.. math::
   Precision = \frac {True Positives (TP)}{True Positives (TP)  + False Positives (FP)}

Interpretation
==============
Precision answers the question: "Of all the instances predicted as positive, how many were actually positive?" A higher precision indicates that a larger proportion of predicted positives are indeed correct, which is particularly important in scenarios where false positives carry significant costs or consequences.

Example
=======
For instance, in a T cell classification task, if a model predicts 100 cells as T cells but only 80 of those are indeed T cells (20 are false positives), the precision would be:

.. math::
   Precision = \frac {80}{80+20} = \frac {80}{100} = 0.8 = 80\%

This means that 80% of the cells classified as T cells were actually T cells.


Recall/Sensitivity
******************
Recall, also known as sensitivity or the true positive rate, is a critical metric in classification tasks that measures the ability of a machine learning model to correctly identify all relevant instances within a dataset. It quantifies how many of the actual positive cases were accurately predicted by the model.Usefull for scAdam model quality control.

Mathematically, recall/sensitivity can be expressed as:

.. math::
   Recall/Sensitivity = \frac {True Positives (TP)}{True Positives (TP) + False Negatives (FN)}

Interpretation
==============
Recall/Sensitivity answers the question: "What fraction of actual positive instances are correctly identified by the model?" It measures the ability of a classification model to capture all relevant instances from the dataset. 

Example
=======
Suppose a T cell detection model is evaluated on a dataset containing 100 actual T cells. The model correctly identified 80 of these T cells and missed 20.

.. math::
   Recall/Sensitivity = \frac {80}{80+20} = \frac {80}{100} = 0.8 = 80\%


F1-score
********
The F1-score is a crucial evaluation metric used in machine learning, particularly for classification tasks. It combines both precision and recall into a single score, providing a balanced measure of a model's performance. This metric is especially useful in situations where the class distribution is imbalanced or when the costs of false positives and false negatives are significant.

Mathematically, f1-score can be expressed as:

.. math::
   F1_score = 2 \times \frac {Precision + Recall}{Precision × Recall}

Interpretation
==============
The F1-score ranges from 0 to 1, where:
* 0 indicates the worst performance (the model failed to identify any true positives).
* 1 indicates perfect precision and recall (the model correctly identifies all positive instances without any false positives).

A high F1 score generally signifies a well-balanced model that achieves both high precision and high recall, while a low F1 score often indicates a trade-off between these two metrics, suggesting that the model struggles to balance them effectively.
​
Example
=======
Suppose we evaluate the performance of a T cell detection model, and we obtain the following metrics:
* Precision: 0.85 (the model correctly identifies 85% of the T cells)
* Recall: 0.75 (the model correctly identifies 75% of all actual T cells)

.. math::
   F1_score = 2 \times \frac {0.85 + 0.75}{0.85 × 0.75} = 0.797 = 79.7\%


Accuracy
********
Accuracy is a fundamental metric used to evaluate the performance of machine learning models, particularly in classification tasks. It measures the overall correctness of a model's predictions by calculating the proportion of correct predictions out of the total number of predictions made.

Mathematically, accuracy can be expressed as:

.. math::
   Accuracy = \frac {Correct Predictions}{Total Predictions} = \frac {TP+TN}{TP+TN+FP+FN}

Typically, scRNA-seq datasets contain many cell types. Therefore, the problem of cell type annotation should be regarded as a multiclass classification problem. In the context of multiclass classification (scRNA-seq cell type anotation), **accuracy** can also be expressed as:

.. math::
   Accuracy = \frac {\epsilon_i=1^N TP_i}{\epsilon_i=1^N (TP_i + FP_i + FN_i)}

​Interpretation
==============
Accuracy values range from 0 to 1, or 0% to 100%. An accuracy of 1 (or 100%) indicates perfect predictions, while an accuracy of 0 means that all predictions were incorrect.

Limitations
===========
While accuracy is a straightforward and intuitive measure, it may not always be the best indicator of model performance, especially in scRNA-seq cell type annotation.

**Accuracy paradox**
The "accuracy paradox" refers to situations where a model achieves high accuracy but performs poorly on critical aspects of the task. This often occurs in scRNA-seq cell type annotation where the majority cell type (CD14+ Monocytes in PBMC) dominates the predictions, leading to misleadingly high accuracy scores while neglecting minority cell types (Innate Lymphoid Cells in PBMC).

To obtain a more comprehensive understanding of model performance, it is essential to use additional metrics such as precision, recall, F1 score, balanced accuracy, and others that account for the specific characteristics of the problem at hand.

Example
=======
Suppose we evaluate the performance of a Monocytes and AXL+ Dendritic cells detection model on a test dataset consisting of 1000 cells. The dataset contains 950 Monocytes and 50 AXL+ Dendritic cells. The model identified that there are 990 Monocytes and 10 AXL+ Dendritic cells in the dataset. Out of the 990 Monocytes identified by the model, 940 are true Monocytes, and out of the 10 AXL+ Dendritic cells, 0 are true AXL+ Dendritic cells. 

.. math::
   Accuracy = \frac {940 + 0}{990 + 10} = \frac {940}{1000} = 0.94 = 94\%

The model has a very high level of accuracy but is unable to detect AXL+ Dendritic cells.
