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

# Create test_rna seurat object with test samples 
Idents(rna) <- 'orig.ident'
test_rna <- subset(rna, idents = c('old2', 'old4', 'oldex2', 'oldex4', 'young1', 'young4'), invert = F) 
table(test_rna$orig.ident)

# Reference dataset ('old1', 'oldex2', 'young2', 'oldex1')
rna <- subset(rna, idents = c('young2', 'old1', 'oldex1'), invert = F)
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

saveRDS(rna, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/Seurat_test/ref.rda')
# Set of folders with 5 test datasets 
# Each test dataset contains 1 donor 
#vec_test <- c('young2', 'old2', 'old4', 'oldex1', 'oldex2')
vec_test <- c('old2', 'old4', 'oldex2', 'oldex4', 'young1', 'young4')

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
      Celltype = "Celltype"
    ),
    reference.reduction = "integrated.cca"
  )
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Mouse_aging_brain/Seurat_test/reports', folder, 'meta.csv'))
}
