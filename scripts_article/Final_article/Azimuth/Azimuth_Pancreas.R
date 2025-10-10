# R 4.2.1
library(Seurat) # 5.1.0
library(Azimuth) # 0.5.0
library(DropletUtils) # 1.18.1

# RNA assay
rna <- Read10X('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/dataset/RNA', gene.column = 1)
rna <- CreateSeuratObject(rna)
meta <- read.csv('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/dataset/metadata.csv', row.names = 1)
rna <- subset(rna, cells = rownames(meta), invert = F)
rna@meta.data <- meta 

# Create test_rna seurat object with test samples 
Idents(rna) <- 'donor_id'
test_rna <- subset(rna, idents = c('HP-20152-01', 'HP17225-01T2D', 'HP19047-01', 'SAMN11867362', 'SAMN12022246', 'SAMN13108021', 'SAMN11642375',
                                   'SAMN11476721', 'SAMN10873960', 'SAMN10439569', 'SAMN10737781', 'SAMN13319813', 'SAMN11522709', 'HP18304', 'SAMN11157311'), invert = F) 
table(test_rna$donor_id)

# Reference dataset
rna <- subset(rna, idents = c('SAMN16365027', 'SAMN17528599'), invert = F)
table(rna$donor_id)

# Split dataset into separate samples
obj_list <- SplitObject(object = rna, split.by = "donor_id")

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
saveRDS(ref_sct, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/ref.rda')


# Create reference (ref) and test objects  
ref_azimuth <- AzimuthReference(
  object = ref_sct,
  refUMAP = "umap",
  refDR = "pca",
  refAssay = "integrated",
  metadata = c("celltype_l1"),
  dims = 1:50,
  k.param = 31,
  reference.version = "1.0.0"
)

# Save Azimuth compatible reference dataset
ref.dir <- file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/reference')
SaveAnnoyIndex(object = ref_azimuth[["refdr.annoy.neighbors"]], file = file.path(ref.dir, "idx.annoy"))
saveRDS(object = ref_azimuth, file = file.path(ref.dir, "ref.Rds"))

# Set of folders with 10 test datasets 
vec_test_1 <- c('HP-20152-01', 'HP17225-01T2D', 'HP19047-01', 'SAMN11867362', 'SAMN12022246')
vec_test_2 <- c('SAMN13108021', 'SAMN11476721', 'SAMN10439569', 'SAMN13319813', 'HP18304')
vec_test_3 <- c('SAMN11642375', 'SAMN10873960', 'SAMN10737781', 'SAMN11522709', 'SAMN11157311')

for (folder in 1:length(vec_test_1)) {
  test <- subset(test_rna, idents = c(vec_test_1[folder]), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     reference = ref.dir)
  
  # Save metadata with real (cell_type) and 
  # predicted (predicted.cell_type) annotation levels  
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/reports', folder))
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/reports', folder, 'meta.csv'))
}

for (folder in 1:length(vec_test_2)) {
  test <- subset(test_rna, idents = c(vec_test_2[folder], vec_test_3[folder]), invert = F)
  
  # Use Azimuth reference to predict cell types in test dataset
  test <- RunAzimuth(test, 
                     reference = ref.dir)
  
  # Save metadata with real (cell_type) and 
  # predicted (predicted.cell_type) annotation levels  
  folder <- folder + 5
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/reports', folder))
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Azimuth_test/reports', folder, 'meta.csv'))
}
