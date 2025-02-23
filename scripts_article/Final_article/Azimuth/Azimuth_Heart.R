# R 4.2.1
library(Seurat) # 5.1.0
library(Azimuth) # 0.5.0
library(DropletUtils) # 1.18.1

# RNA assay
meta <- read.csv('C:/Users/vadim/scRNA/Heart_CITE/metadata.csv', row.names = 1)
rna <- Read10X('C:/Users/vadim/scRNA/Heart_CITE/RNA/', gene.column = 1)
rna <- CreateSeuratObject(rna, meta.data = meta)

# Create test_rna seurat object with test samples (all except zero day samples)
Idents(rna) <- 'orig.ident'
test_rna <- subset(rna, idents = c(12, 13, 17, 27, 28, 29, 30, 32, 39, 42), invert = T)
table(test_rna$orig.ident)

# Leave only zero day samples
rna <- subset(rna, idents = c(12, 13, 17, 27, 28, 29, 30, 32, 39, 42), invert = F)
table(rna$orig.ident)

# Split dataset into separate samples
obj_list <- SplitObject(object = rna, split.by = "orig.ident")

# https://github.com/satijalab/azimuth-references/blob/master/human_motorcortex/scripts/integrate.R
# SCTransform of separate samples
obj_list <- lapply(X = obj_list, FUN = SCTransform, method = "glmGamPoi")

# Select 5000 integration features and prepare data for integration
features <- SelectIntegrationFeatures(object.list = obj_list, nfeatures = 3000)
obj_list <- PrepSCTIntegration(object.list = obj_list, anchor.features = features)

# PCA for each sample
obj_list <- lapply(X = obj_list, FUN = RunPCA, features = features)

# Perform integration 
# Find integration anchors using Reciprocal PCA method and SCTransform normalized data
anchors <- FindIntegrationAnchors(
  object.list = obj_list,
  normalization.method = "SCT",
  anchor.features = features,
  dims = 1:30,
  reduction = "rpca",
  k.anchor = 5
)

# Integrate data using pre-computed anchors
ref_sct <- IntegrateData(
  anchorset = anchors,
  normalization.method = "SCT",
  dims = 1:30, 
  new.assay.name = "integrated"
)

# PCA for integrated assay
ref_sct <- RunPCA(ref_sct)
ElbowPlot(ref_sct, ndims = 50)

# UMAP dimensional reduction with uwot model (used in Azimuth reference construction)
ref_sct <- RunUMAP(
  object = ref_sct,
  reduction = "pca",
  dims = 1:20,
  return.model = TRUE
)
saveRDS(ref_sct, 'C:/Users/vadim/scRNA/Heart_CITE/ref_sct.rda')

# ADT
# Load data and subset based on meta_modified from python script PBMC_3p_dataset
adt <- Read10X('C:/Users/vadim/scRNA/Heart_CITE/ADT/', gene.column = 1)
adt <- CreateSeuratObject(adt, meta.data = meta)

# Normalize protein data using the same normalization method (CLR) as in mdata object in python
adt <- NormalizeData(
  adt,
  normalization.method = "CLR",
  margin = 1,
)

# Create test_adt assay with test samples (all except zero day samples)
Idents(adt) <- 'orig.ident'
test_adt <- subset(adt, idents = c(12, 13, 17, 27, 28, 29, 30, 32, 39, 42), invert = T)
test_adt <- GetAssayData(test_adt)
# Create test_adt assay 
test_adt <- CreateAssayObject(test_adt)

# Leave only zero day samples  for reference dataset
adt <- subset(adt, idents = c(12, 13, 17, 27, 28, 29, 30, 32, 39, 42), invert = F)
adt <- GetAssayData(adt)
# Create adt assay 
adt <- CreateAssayObject(adt)

# Add adt assay to reference
ref_sct[["ADT"]] <- adt
# Add test_adt assay to test_rna
test_rna[['ADT']] <- test_adt

# Save integrated dataset
saveRDS(ref_sct, 'C:/Users/vadim/scRNA/Heart_CITE/ref_sct.rda')

# Create reference (ref) and test objects  

ref_azimuth <- AzimuthReference(
  object = ref_sct,
  refUMAP = "umap",
  refDR = "pca",
  refAssay = "integrated",
  metadata = c("celltype_l1", "celltype_l2"),
  assays = 'ADT',
  dims = 1:50,
  k.param = 31,
  reference.version = "1.0.0"
)

# Save Azimuth compatible reference dataset
ref.dir <- file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth/reference')
SaveAnnoyIndex(object = ref_azimuth[["refdr.annoy.neighbors"]], file = file.path(ref.dir, "idx.annoy"))
saveRDS(object = ref_azimuth, file = file.path(ref.dir, "ref.Rds"))

# Set of folders with 8 test datasets 
vec_test_1 <- c('1_6', '2_7', '4_5', '8_9')
vec_test_2 <- c('41', '34', '15', '33')

for (folder in vec_test_1) {
  # Sample A
  A <- as.integer(substr(folder, start = 1, stop = 1))
  # Sample B
  B <- as.integer(substr(folder, start = 3, stop = 3))
  test <- subset(test_rna, idents = c(A, B), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     do.adt = T,
                     reference = ref.dir)
  
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', paste(A,B, sep = '_')))
  # Save metadata with real (celltype_l1, celltype_l2) and 
  # predicted (predicted.celltype_l1, predicted.celltype_l2) annotation levels  
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', paste(A,B, sep = '_'), 'meta.csv'))
  
  # Save real adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', paste(A,B, sep = '_'), 'ADT_matrix.h5'),
    test@assays[["ADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
  # Save imputed adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', paste(A,B, sep = '_'), 'impADT_matrix.h5'),
    test@assays[["impADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
}

for (folder in vec_test_2) {
  A <- as.integer(folder)
  test <- subset(test_rna, idents = c(A), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     do.adt = T,
                     reference = ref.dir)
  
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', A))
  # Save metadata with real (celltype_l1, celltype_l2) and 
  # predicted (predicted.celltype_l1, predicted.celltype_l2) annotation levels  
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', A, 'meta.csv'))
  
  # Save real adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', A, 'ADT_matrix.h5'),
    test@assays[["ADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
  # Save imputed adt normalized data for future comparison by scparadise.scnoah.report_reg() in python
  write10xCounts(
    path = file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Heart_3p_CITE/Azimuth', A, 'impADT_matrix.h5'),
    test@assays[["impADT"]]@data,
    type = 'HDF5',
    overwrite = TRUE
  )
}
