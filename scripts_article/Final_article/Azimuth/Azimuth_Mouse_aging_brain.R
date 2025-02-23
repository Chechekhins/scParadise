# R 4.2.1
library(Seurat) # 5.1.0
library(Azimuth) # 0.5.0
library(DropletUtils) # 1.18.1

# RNA assay
rna <- Read10X('C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/dataset/')
rna <- CreateSeuratObject(rna)
meta <- read.csv('C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/meta.tsv', sep = '\t', row.names = 1)
rna <- subset(rna, cells = rownames(meta), invert = F)
rna@meta.data <- meta 

# Create test_rna seurat object with test samples ('old1', 'oldex4', 'young1', 'young4')
Idents(rna) <- 'orig.ident'
test_rna <- subset(rna, idents = c('old1', 'oldex1', 'oldex2', 'young2'), invert = T)
table(test_rna$orig.ident)

# Leave only zero day samples
rna <- subset(rna, idents = c('old1', 'oldex1', 'oldex2', 'young2'), invert = F)
table(rna$orig.ident)

# Split dataset into separate samples
obj_list <- SplitObject(object = rna, split.by = "orig.ident")

# https://github.com/satijalab/azimuth-references/blob/master/human_motorcortex/scripts/integrate.R
# SCTransform of separate samples
obj_list <- lapply(X = obj_list, FUN = SCTransform, method = "glmGamPoi")

# Select 3000 integration features and prepare data for integration
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

# Save integrated dataset
saveRDS(ref_sct, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/ref_sct.rda')

# Create Azimuth reference
ref_azimuth <- AzimuthReference(
  object = ref_sct,
  refUMAP = "umap",
  refDR = "pca",
  refAssay = "integrated",
  metadata = c("Celltype"),
  dims = 1:50,
  k.param = 31,
  reference.version = "1.0.0"
)

# Save Azimuth compatible reference dataset
ref.dir <- file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/Azimuth/reference')
SaveAnnoyIndex(object = ref_azimuth[["refdr.annoy.neighbors"]], file = file.path(ref.dir, "idx.annoy"))
saveRDS(object = ref_azimuth, file = file.path(ref.dir, "ref.Rds"))

# Set of folders with 8 test datasets 
# Each test dataset contains 2 separate donors 
vec_test <- c('young4', 'old2', 'old4', 'young1', 'oldex4')

for (folder in vec_test) {
  test <- subset(test_rna, idents = c(folder), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     reference = ref.dir)
  
  # Save metadata with real (celltype_l1, celltype_l2, celltype_l3) and 
  # predicted (predicted.celltype_l1, predicted.celltype_l2, predicted.celltype_l3) annotation levels  
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/Azimuth/reports', folder, 'meta.csv'))
}
