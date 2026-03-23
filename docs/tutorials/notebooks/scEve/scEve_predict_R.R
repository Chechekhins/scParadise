install.packages('Seurat')
install.packages('reticulate')
devtools::install_github("cellgeni/sceasy")
BiocManager::install("scDblFinder")

# import R libraries
library(Seurat)
library(scDblFinder)
library(sceasy)
library(reticulate)
set.seed(0)

# import Python libraries
use_condaenv('scparadise')
sc <- import("scanpy", convert = FALSE)
scp <- import("scparadise", convert = FALSE)

## Download dataset from 10x Genomics
# Specify the URL of the file you want to download
url <- "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_X/10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
# Specify the file name and location where you want to save the file on your computer
file_name <- "10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
file_path <- ""
# Call the download.file() function, passing in the URL and file name/location as arguments
# Or just paste the url to a browser and download it directly
download.file(url, paste(file_path, file_name, sep = ""), mode = "wb")

# Load dataset using file name/location
object <- Read10X_h5(paste(file_path, file_name, sep = ""))
# Or read your Seurat object:
# object <- readRDS('object.rds')

## Preprocessing 
# Find doublets using scDblFinder
sce <- scDblFinder(object)

# Create Seurat object + add scDblFinder results
object <- CreateSeuratObject(object, min.cells = 3)
object$scDblFinder.class <- sce$scDblFinder.class
object$scDblFinder.score <- sce$scDblFinder.score
rm(sce)

# Visualize number of genes, counts, percent of mito genes in cells
object[["percent.mt"]] <- PercentageFeatureSet(object, pattern = "^MT-")

VlnPlot(
  object,
  features = c("nFeature_RNA", 
               "nCount_RNA", 
               "percent.mt"),
  ncol = 3,
  group.by = 'scDblFinder.class'
)

## Remove low quality cells
object <-
  subset(object, subset = 
           nFeature_RNA > 200 &
           nFeature_RNA < 5000 & 
           nCount_RNA < 20000 & 
           percent.mt < 15)
## Remove doublets
object <-
  subset(object, scDblFinder.class == 'singlet')

# Data normalization, PCA
DefaultAssay(object) <- 'RNA'
object <- SCTransform(object, vst.flavor = 'v2', vars.to.regress = "percent.mt") |> 
  FindVariableFeatures() |> 
  ScaleData() |> 
  RunPCA() |> 
  RunUMAP(dims = 1:15)

# Create Assay of Seurat v4 structure 
# You can skip this step if your object is in Seurat v4 format
object[["RNA_convert"]] <- as(object = object@assays$RNA, Class = "Assay")

# Convert Seurat object (R) to AnnData (Python)
adata <- convertFormat(object, assay = 'RNA_convert', from="seurat", to="anndata", main_layer="counts", drop_single_values=FALSE)
adata <- adata$copy()

# Remove 'RNA_convert' assay
object[["RNA_convert"]] <- NULL

# Normalizing to median total counts
sc$pp$normalize_total(adata, target_sum = NULL) #, target_sum = FALSE)
# Logarithmize the data
sc$pp$log1p(adata)

# Download dataframe with available models 
df <- scp$sceve$available_models()
View(df)

## scEve 
# Download scEve model
scp$sceve$download_model(model_name = 'Human_PBMC_3p', save_path = '')

# Impute surface proteins using scEve model
adata_adt_imp <- scp$sceve$predict(adata, path_model = 'Human_PBMC_3p_scEve', return_mdata = FALSE)

## scAdam (you can skip this step if cells are already annotated)
# Download scAdam model
scp$scadam$download_model(model_name = 'Human_PBMC', save_path = '')

# Predict cell types using scAdam model
scp$scadam$predict(adata, path_model = 'Human_PBMC_scAdam')

# Add AnnData.obs to Seurat object meta.data
meta <- py_to_r(adata$obs)
object@meta.data <- meta

## Convert AnnData adata_adt_pred to dgCMatrix adt_pred
adt_imp <- t(as.sparse(py_to_r(adata_adt_imp$X$toarray())))
adt_imp@Dimnames[[1]] <- py_to_r(adata_adt_imp$var_names$tolist())
adt_imp@Dimnames[[2]] <- py_to_r(adata_adt_imp$obs_names$tolist())
rm(adata_adt_imp)

# Create ADT assay in Seurat object
adt_imp <- CreateAssay5Object(counts = adt_imp, data = adt_imp)
object@assays$ADT <- adt_imp
object@assays[["ADT"]]@key <- 'adt_'

# PCA on imputed ADT data
DefaultAssay(object) <- 'ADT'
VariableFeatures(object) <- rownames(object[["ADT"]])
# scEve predicts normalized data, so we skip normalization step
object <- FindVariableFeatures(object,  nfeatures = 100) |> 
  ScaleData() |> 
  RunPCA(reduction.name = 'apca')

ElbowPlot(object, ndims = 50, reduction = 'pca')
ElbowPlot(object, ndims = 50, reduction = 'apca')

# Construct weighted nearest neighbor graph
object <- FindMultiModalNeighbors(
  object, reduction.list = list("pca", "apca"), 
  dims.list = list(1:15, 1:15)
)

# UMAP dimensional reduction using WNN nn 
object <-
  RunUMAP(
    object,
    nn.name = "weighted.nn",
    reduction.name = "wnn.umap",
    reduction.key = "wnnUMAP_"
  )

## Visualize predictions on UMAP
# Celltype_l1
p1 <- DimPlot(
  object,
  group.by = c('pred_celltype_l1'),
  reduction = 'wnn.umap', # WNN UMAP
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("WNN UMAP")
p2 <- DimPlot(
  object,
  group.by = c('pred_celltype_l1'),
  reduction = 'umap',  # UMAP calculated using only gene expression data
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("RNA UMAP")
p1+p2

# Celltype_l2
p1 <- DimPlot(
  object,
  group.by = c('pred_celltype_l2'),
  reduction = 'wnn.umap', # WNN UMAP
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("WNN UMAP")
p2 <- DimPlot(
  object,
  group.by = c('pred_celltype_l2'),
  reduction = 'umap',  # UMAP calculated using only gene expression data
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("RNA UMAP")
p1+p2

# Celltype_l3
p1 <- DimPlot(
  object,
  group.by = c('pred_celltype_l3'),
  reduction = 'wnn.umap', # WNN UMAP
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("WNN UMAP")
p2 <- DimPlot(
  object,
  group.by = c('pred_celltype_l3'),
  reduction = 'umap',  # UMAP calculated using only gene expression data
  label = T,
  repel = T
) + NoLegend() + ggplot2::ggtitle("RNA UMAP")
p1+p2

## Check predictions and imputations
# CD4+ T cells
FeaturePlot(
  object,
  features = c('CD3E', # RNA
               'CD4', # RNA
               'adt-CD3-2-pred', # ADT
               'adt-CD4-1-pred' # ADT
               ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
)

# CD8+ T cells
FeaturePlot(
  object,
  features = c('CD3E', # RNA
               'CD8A', # RNA
               'adt-CD3-2-pred', # ADT
               'adt-CD8a-pred' # ADT
  ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
)

# CD14+ Monocytes
FeaturePlot(
  object,
  features = c('CD14', # RNA
               'LYZ', # RNA
               'adt-CD14-pred', # ADT
               'adt-CD11c-pred' # ADT
               ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
)

# CD16+ Monocytes
FeaturePlot(
  object,
  features = c('FCGR3A', # RNA
               'MS4A7', # RNA
               'adt-CD16-pred', # ADT
               'adt-CX3CR1-pred' # ADT
  ),
  max.cutoff = 'q99',
  min.cutoff = 'q01',
  order = T
)

# B
FeaturePlot(
  object,
  features = c('CD79A', # RNA
               'BANK1', # RNA
               'adt-CD19-pred', # ADT
               'adt-CD20-pred' # ADT
  ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
) 

# NK
FeaturePlot(
  object,
  features = c('KLRF1', # RNA
               'FCER1G', # RNA
               'adt-CD56-1-pred', # ADT
               'adt-CD56-2-pred' # ADT
  ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
) 

# Platelet
FeaturePlot(
  object,
  features = c('PF4', # RNA
               'PPBP', # RNA
               'adt-CD42b-pred', # ADT
               'adt-CD62P-pred' # ADT
  ),
  order = T,
  max.cutoff = 'q99',
  min.cutoff = 'q01',
)

