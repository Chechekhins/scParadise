# R 4.2.1
library(Seurat) # 5.1.0
library(Azimuth) # 0.5.0
library(DropletUtils) # 1.18.1

# RNA assay
rna <- Read10X('PBMC_ref/CITEseq/RNA/')
rna <- CreateSeuratObject(rna)
meta <- read.csv('PBMC_ref/CITEseq/meta_modified.csv', row.names = 1)
rna <- subset(rna, cells = rownames(meta), invert = F)
rna@meta.data <- meta 

# Create test_rna seurat object with test samples (all except zero day samples)
Idents(rna) <- 'orig.ident'
test_rna <- subset(rna, idents = c('P1_0','P2_0','P3_0','P4_0','P5_0','P6_0','P7_0','P8_0'), invert = T)
table(test_rna$orig.ident)

# For comparison with scAdam we used the same dataset to create reference: 'P8_0', 'P7_0', 'P2_0', 'P6_0'
# Below we used datasets for comparison with scEve
rna <- subset(rna, idents = c('P1_0','P2_0','P3_0','P4_0','P5_0','P6_0','P7_0','P8_0'), invert = F)
table(rna$orig.ident)

# Split dataset into separate samples
rna[["RNA"]] <- split(rna[["RNA"]], f = rna$orig.ident)
# Normalize, HVG, Scale, PCA
rna <- NormalizeData(rna)
rna <- FindVariableFeatures(rna)
rna <- ScaleData(rna)
rna <- RunPCA(rna)

# Perform integration 
rna <- IntegrateLayers(object = rna, 
                       method = CCAIntegration, 
                       orig.reduction = "pca",
                       new.reduction = "integrated.cca", 
                       verbose = FALSE)

saveRDS(rna, 'C:/Users/vadim/scRNA/scParadise/scripts_article/PBMC_3p_CITE/Seurat_test/ref_test.rda')

# ADT
# Load data and subset based on meta_modified from python script PBMC_3p_dataset
adt <- Read10X('PBMC_ref/CITEseq/ADT/')
adt <- CreateSeuratObject(adt)
adt <- subset(adt, cells = rownames(meta), invert = F)
adt@meta.data <- meta 

adt <- GetAssayData(adt)
prots <- adt@Dimnames[[1]]
isotype_controls <- c("Rat-IgG1-1", "Rat-IgG2b", "Rat-IgG1-2", "Rag-IgG2c")
prots.use <- prots[! prots %in% isotype_controls]
adt <- adt[prots.use, ]

# Normalize protein data using the same normalization method (CLR) as in mdata object in python
adt <- CreateSeuratObject(adt, meta.data = meta)
adt <- NormalizeData(
  adt,
  normalization.method = "CLR",
  margin = 1,
)

# Create test_adt assay with test samples (all except zero day samples)
Idents(adt) <- 'orig.ident'
test_adt <- subset(adt, idents = c('P1_0','P2_0','P3_0','P4_0','P5_0','P6_0','P7_0','P8_0'), invert = T)
test_adt <- GetAssayData(test_adt)
# Create test_adt assay 
test_adt <- CreateAssayObject(test_adt)

# For comparison with scAdam we used the same dataset to create reference: 'P8_0', 'P7_0', 'P2_0', 'P6_0'
# Below we used datasets for comparison with scEve
adt <- subset(adt, idents = c('P1_0', 'P2_0', 'P3_0', 'P4_0', 'P5_0', 'P6_0', 'P7_0', 'P8_0'), invert = F)
adt <- GetAssayData(adt)
# Create adt assay 
adt <- CreateAssayObject(adt)

# Add adt assay to reference
rna[["ADT"]] <- adt
# Add test_adt assay to test_rna
test_rna[['ADT']] <- test_adt

saveRDS(rna, 'C:/Users/vadim/scRNA/scParadise/scripts_article/PBMC_3p_CITE/Seurat_test/ref_test.rda')

# Set of folders with 8 test datasets 
# Each test dataset contains 2 separate donors 
vec_test <- c('P1_3_P3_3', 'P1_7_P8_3', 'P2_3_P4_7', 'P2_7_P6_3', 'P3_7_P7_3', 'P4_3_P7_7', 'P5_3_P8_7', 'P5_7_P6_7')

for (folder in vec_test) {
  # Sample A (P1_0)
  A <- substr(folder, start = 1, stop = 4)
  # Sample B (P4_3)
  B <- substr(folder, start = 6, stop = 9)
  test <- subset(test_rna, idents = c(A, B), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- NormalizeData(test)
  
  anchors <- FindTransferAnchors(
    reference = rna,
    query = test,
    dims = 1:30,
    reference.reduction = "integrated.cca"
  )
  test <- MapQuery(
    anchorset = anchors,
    query = test,
    reference = rna,
    refdata = list(
      celltype.l1 = "celltype_l1",
      celltype.l2 = "celltype_l2",
      celltype.l3 = "celltype_l3",
      impADT = "ADT"
    ),
    reference.reduction = "integrated.cca"
  )
  
  # Save metadata with real (celltype_l1, celltype_l2, celltype_l3) and 
  # predicted (predicted.celltype_l1, predicted.celltype_l2, predicted.celltype_l3) annotation levels  
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/PBMC_3p_CITE/Seurat_test/reports', paste(A,B, sep = '_'), 'meta.csv'))
  
  # Save real adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/PBMC_3p_CITE/Seurat_test/reports', paste(A,B, sep = '_'), 'ADT_matrix.h5'),
    test@assays[["ADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
  # Save imputed adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/PBMC_3p_CITE/Seurat_test/reports', paste(A,B, sep = '_'), 'impADT_matrix.h5'),
    test@assays[["impADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
}
