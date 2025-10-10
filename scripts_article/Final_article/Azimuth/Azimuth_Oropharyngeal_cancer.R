# R 4.2.1
library(Seurat) # 5.1.0
library(Azimuth) # 0.5.0
library(DropletUtils) # 1.18.1

# RNA assay
rna <- Read10X('C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/RNA', gene.column = 1)
rna <- CreateSeuratObject(rna)
meta <- read.csv('C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/metadata.csv', sep = ',', row.names = 1)
rna@meta.data <- meta 

# Create test_rna seurat object with test samples 
Idents(rna) <- 'donor_id'
test_rna <- subset(rna, idents = c('HN490', 'HN492', 'HN482', 'HN488', 'HN485', 'HN487', 'HN483', 'HN489'), invert = F) 
table(test_rna$donor_id)

# Reference dataset
rna <- subset(rna, idents = c('HN481'), invert = F)
table(rna$donor_id)
table(rna$sample_id)

# Split dataset into separate samples
obj_list <- SplitObject(object = rna, split.by = "sample_id")

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
saveRDS(ref_sct, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/Azimuth/ref.rda')


# Create reference (ref) and test objects  
ref_azimuth <- AzimuthReference(
  object = ref_sct,
  refUMAP = "umap",
  refDR = "pca",
  refAssay = "integrated",
  metadata = c("cell_type"),
  dims = 1:50,
  k.param = 31,
  reference.version = "1.0.0"
)

# Save Azimuth compatible reference dataset
ref.dir <- file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/Azimuth/ref_azimuth')
SaveAnnoyIndex(object = ref_azimuth[["refdr.annoy.neighbors"]], file = file.path(ref.dir, "idx.annoy"))
saveRDS(object = ref_azimuth, file = file.path(ref.dir, "ref.Rds"))

# Set of folders with 8 test datasets 
# Each test dataset contains 2 separate donors 
vec_test <- c('HN490', 'HN492', 'HN482', 'HN488', 'HN485', 'HN487', 'HN483', 'HN489')

for (folder in vec_test) {
  # Sample
  test <- subset(test_rna, idents = c(folder), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     reference = ref.dir)
  
  # Save metadata with real (cell_type) and 
  # predicted (predicted.cell_type) annotation levels  
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/Azimuth/reports', folder, 'meta.csv'))
}
