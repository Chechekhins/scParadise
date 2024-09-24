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

## Download dataset frpm 10x Genomics
# Specify the URL of the file you want to download
url <- "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_X/10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
# Specify the file name and location where you want to save the file on your computer
file_name <- "10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
file_path <- ""
# Call the download.file() function, passing in the URL and file name/location as arguments
download.file(url, paste(file_path, file_name, sep = ""), mode = "wb")

# Load dataset using file name/location
object <- Read10X_h5(paste(file_path, file_name, sep = ""))

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

## Subset low quality cells
object <-
  subset(object, subset = 
           nFeature_RNA > 200 &
           nFeature_RNA < 5000 & 
           nCount_RNA < 20000 & 
           percent.mt < 15)
object <-
  subset(object, scDblFinder.class == 'singlet')

# Data normalization, PCA, UMAP
object <- object |> 
  SCTransform(vst.flavor = 'v2', vars.to.regress = "percent.mt") |>
  RunPCA() |> 
  RunUMAP(dims = 1:30)

## Conversion of Seurat object to AnnData and prediction using scAdam model
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
# Set the .raw attribute of the AnnData object 
# to the normalized and logarithmized gene expression 
# for later use by scparadise
adata$raw <- adata

# Download dataframe with available models 
df <- scp$scadam$available_models()
View(df)

# Download scAdam model
scp$scadam$download_model(model_name = 'PBMC', save_path = '')

# Predict cell types using scAdam model
scp$scadam$predict(adata, path_model = 'PBMC')

# Add AnnData.obs to Seurat object meta.data
meta <- py_to_r(adata$obs)
object@meta.data <- meta

## Visualize predictions on UMAP
# Celltype_l1
DimPlot(object,
        group.by = c('pred_celltype_l1'),
        pt.size = 2,
        label = T,
        label.size = 5, 
        #cols = colorspace::qualitative_hcl(n = 16, l = 55, l1 = 55, c1 = 200),
        repel = T) + NoLegend()

# Celltype_l2
DimPlot(object,
        group.by = c('pred_celltype_l2'),
        pt.size = 2,
        label = T,
        label.size = 4, 
        #cols = colorspace::qualitative_hcl(n = 16, l = 55, l1 = 55, c1 = 200),
        repel = T) + NoLegend()

# Celltype_l3
DimPlot(object,
        group.by = c('pred_celltype_l3'),
        pt.size = 2,
        label = T,
        label.size = 3, 
        #cols = colorspace::qualitative_hcl(n = 16, l = 55, l1 = 55, c1 = 200),
        repel = T) + NoLegend()

## Check prediction
# T cells
FeaturePlot(object,
            features = c('CD3E', 'CD3D', # pan-T markers
                         'CD4', # CD4+ T
                         'CD8A' # CD8+ T
                         ),
            order = T,
            pt.size = 1)

# Monocytes
FeaturePlot(object,
            features = c('CD14', 'LYZ', # CD14 Mono
                         'FCGR3A', 'MS4A7' # CD16 Mono
                         ),
            order = T,
            pt.size = 1)

# B
FeaturePlot(object,
            features = c('SSPN', 'IGHA2', # B memory
                         'TCL1A', 'IGHD' # B naive
            ),
            order = T,
            pt.size = 1) 

# NK
FeaturePlot(object,
            features = c('KLRF1', 'FCER1G', # NK
                         'XCL1', 'NCAM1' # NK CD56bright
            ),
            order = T,
            pt.size = 1) 

# HSPC, ASDC
FeaturePlot(object,
            features = c('PRSS57', 'CD34', # HSPC
                         'AXL', 'PPP1R14A' # ASDC
            ),
            order = T,
            pt.size = 1)

