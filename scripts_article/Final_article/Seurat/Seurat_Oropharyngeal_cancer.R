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
rna[["RNA"]] <- split(rna[["RNA"]], f = rna$sample_id)

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

saveRDS(rna, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/Seurat/ref.rds')
# Set of folders with 8 test datasets 
# Each test dataset contains 1 donor 
vec_test <- c('HN490', 'HN492', 'HN482', 'HN488', 'HN485', 'HN487', 'HN483', 'HN489')

for (folder in vec_test) {
  test <- subset(test_rna, idents = c(folder), invert = F)
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
      cell_type = "cell_type"
    ),
    reference.reduction = "integrated.cca"
  )
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Oropharyngeal_cancer/Seurat/reports', folder, 'meta.csv'))
}
