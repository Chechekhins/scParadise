Theory
######
This section aims to clarify the key concepts that underpin the operation of the scParadise library. Below are the characteristics of the scAdam and scEve models. Additionally, the quality control metrics used by scNoah are explained.


.. math::
   Precision = 
   \frac 
               {
               True Positives (TP)
               } {True Positives (TP)  + False Positives (FP)}

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


scNoah mrtrics
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


Precision
*********
Precision is a key metric in machine learning that evaluates the accuracy of a model's positive predictions. It is defined as the ratio of true positive predictions to the total number of instances predicted as positive (which includes both true positives and false positives). Usefull for scAdam model quality control.

Mathematically, precision can be expressed as:

.. math::
   Precision = True Positives (TP) \\ True Positives (TP)  + False Positives (FP)

Where:
•	True Positives (TP): The number of correct positive predictions made by the model.
•	False Positives (FP): The number of incorrect positive predictions made by the model.

