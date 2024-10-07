scParadise models
=================

scParadise includes 2 type of models:

1) scAdam models - reference free fast multi-level multi-class cell type annotation.
2) scEve models - reference free fast modality prediction in scRNA-seq data.

scAdam models
-------------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model
     - Description
     - Suspension
     - Accuracy
     - Balanced Accuracy
     - Number of Levels
     - Reference
   * - Human_PBMC
     - Peripheral blood mononuclear cells of healthy adults 3' scRNA seq	
     - cells
     - 0.979
     - 0.979
     - 3
     - :cite:p:`hao2021integrated‎`
   * - Human_BMMC
     - Bone marrow mononuclear cell of healthy adults
     - cells
     - 0.947
     - 0.942
     - 3
     - :cite:p:`‎NEURIPS_DATASETS_AND_BENCHMARKS2021_158f3069`
   * - Human_Lung
     - Core Human Lung Cell Atlas
     - cells
     - 0.965
     - 0.964
     - 5
     - :cite:p:`‎Sikkema2023`
   * - Human_Retina
     - Single cell atlas of the human retina
     - cells
     - 0.984
     - 0.979
     - 4
     - :cite:p:`li2023integrated‎`
   * - Mouse_Retina
     - Single cell atlas of the mouse retina
     - cells
     - 0.967
     - 0.960
     - 4
     - :cite:p:`‎li2024comprehensive`
   * - Mouse_Cerebellum
     - Single nuclei atlas of the Mouse cerebellum
     - nuclei
     - 0.999
     - 0.999
     - 2
     - :cite:p:`‎kozareva2021transcriptomic,hao2024cross`
   * - Macaque_Cerebellum
     - Single nuclei atlas of the Macaque cerebellum
     - nuclei
     - 0.995
     - 0.994
     - 2
     - :cite:p:`hao2024cross‎`
   * - Marmoset_Cerebellum
     - Single nuclei atlas of the Marmoset cerebellum
     - nuclei
     - 0.988
     - 0.987
     - 2
     - :cite:p:`hao2024cross‎`

scEve models
------------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Model
     - Description
     - Suspension
     - RMSE
     - MAE
     - Number of Proteins
     - Reference
   * - Human_PBMC
     - Peripheral blood mononuclear cells of healthy adults 3' scRNA seq	
     - cells
     - 0.305
     - 0.226
     - 224
     - :cite:p:`hao2021integrated‎`
   * - Human_BMMC
     - Bone marrow mononuclear cell of healthy adults
     - cells
     - 0.706
     - 0.454
     - 134
     - :cite:p:`‎NEURIPS_DATASETS_AND_BENCHMARKS2021_158f3069`

References
----------

.. bibliography:: /refs.bib
   :style: plain
