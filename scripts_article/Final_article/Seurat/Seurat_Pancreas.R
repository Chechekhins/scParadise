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
rna[["RNA"]] <- split(rna[["RNA"]], f = rna$donor_id)
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

saveRDS(rna, 'C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Seurat_test/ref.rda')
# Set of folders test datasets 
vec_test_1 <- c('HP-20152-01', 'HP17225-01T2D', 'HP19047-01', 'SAMN11867362', 'SAMN12022246')
vec_test_2 <- c('SAMN13108021', 'SAMN11476721', 'SAMN10439569', 'SAMN13319813', 'HP18304')
vec_test_3 <- c('SAMN11642375', 'SAMN10873960', 'SAMN10737781', 'SAMN11522709', 'SAMN11157311')

for (folder in 1:length(vec_test_1)) {
  test <- subset(test_rna, idents = c(vec_test_1[folder]), invert = F)
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
      celltype_l1 = "celltype_l1"
    ),
    reference.reduction = "integrated.cca"
  )
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Seurat_test/reports', folder))
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Seurat_test/reports', folder, 'meta.csv'))
}

for (folder in 1:length(vec_test_2)) {
  test <- subset(test_rna, idents = c(vec_test_2[folder], vec_test_3[folder]), invert = F)
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
      celltype_l1 = "celltype_l1"
    ),
    reference.reduction = "integrated.cca"
  )
  folder <- folder + 5
  dir.create(file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Seurat_test/reports', folder))
  write.csv(test@meta.data, file.path('C:/Users/vadim/scRNA/scParadise/scripts_article/Pancreas/Seurat_test/reports', folder, 'meta.csv'))
}
