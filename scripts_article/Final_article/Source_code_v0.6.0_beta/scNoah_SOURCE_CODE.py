import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import RandomOverSampler
import sklearn.metrics as metrics_
from imblearn import metrics
from scipy import sparse
from scipy import stats
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import shap
import json
import anndata
import os



# Function for balance cell types in adata
def balance(
    adata,
    sample = None,
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    shrinkage = 1
):
    '''
    Balance cell types in AnnData object.
    Returns adata_balanced with updated matrix and adata_balanced.obs with given celltypes levels and sample.
    If you give counts function returns counts.
    If you give normalized data function returns normalized data.  
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Function uses adata.X for oversample.
    sample : str, (default: None)
        Samples names key in adata.obs dataframe.
    celltype_l1 : str, (default: None)
        First level of cell annotation. Key in adata.obs dataframe.
    celltype_l2 : str, (default: None)
        Second level of cell annotation. Key in adata.obs dataframe.
    celltype_l3 : str, (default: None)
        Third level of cell annotation. Key in adata.obs dataframe.
    celltype_l4 : str, (default: None)
        Forth level of cell annotation. Key in adata.obs dataframe.
    celltype_l5 : str, (default: None)
        Fifth level of cell annotation. Key in adata.obs dataframe.
    shrinkage : float, (default: 1.0)
        Parameter controlling the shrinkage applied to the covariance matrix 
        when a smoothed bootstrap is generated. 
    
    '''
    
    # Create a dictionary_undersample and dictionary_oversample with number of cells for each cell type in most detailed cell type annotation
    if celltype_l5:
        lst_types = list(np.unique(adata.obs[celltype_l5]))
        dictionary_adata = dict(adata.obs[celltype_l5].value_counts())
    elif celltype_l4:
        lst_types = list(np.unique(adata.obs[celltype_l4]))
        dictionary_adata = dict(adata.obs[celltype_l4].value_counts())
    elif celltype_l3:
        lst_types = list(np.unique(adata.obs[celltype_l3]))
        dictionary_adata = dict(adata.obs[celltype_l3].value_counts())
    elif celltype_l2:
        lst_types = list(np.unique(adata.obs[celltype_l2]))
        dictionary_adata = dict(adata.obs[celltype_l2].value_counts())
    elif celltype_l1:
        lst_types = list(np.unique(adata.obs[celltype_l1]))
        dictionary_adata = dict(adata.obs[celltype_l1].value_counts())
    ## average number of cells in adata most detailed cell type annotation
    cells = int(len(adata)/len(lst_types) + 1)
    dictionary_undersample = {}
    dictionary_oversample = {}
    for i in dictionary_adata.keys():
        if dictionary_adata[i] > cells:
            dictionary_undersample[i] = cells
        else:
            dictionary_oversample[i] = cells

    # Undersample
    
    # Split 'adata' to 'adata_temp' with cell types to subset and 'adata_leave' with cell types to leave without subset
    lst = []
    if celltype_l5:
        adata_temp = adata[adata.obs[celltype_l5].isin(list(dictionary_undersample.keys()))].copy()
        # Subset 'adata_temp' using dictionary
        for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype_l5).indices.items():
            lst.append(np.random.choice(cell_type_indices, dictionary_undersample[cell_type], replace=False))
        annotation = np.unique(adata.obs[celltype_l5]).tolist()
        for i in list(dictionary_undersample.keys()):
            if i in annotation:
                annotation.remove(i)
        adata_leave = adata[adata.obs[celltype_l5].isin(annotation)].copy()
    elif celltype_l4:
        adata_temp = adata[adata.obs[celltype_l4].isin(list(dictionary_undersample.keys()))].copy()
        # Subset 'adata_temp' using dictionary
        for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype_l4).indices.items():
            lst.append(np.random.choice(cell_type_indices, dictionary_undersample[cell_type], replace=False))
        annotation = np.unique(adata.obs[celltype_l4]).tolist()
        for i in list(dictionary_undersample.keys()):
            if i in annotation:
                annotation.remove(i)
        adata_leave = adata[adata.obs[celltype_l4].isin(annotation)].copy()
    elif celltype_l3:
        adata_temp = adata[adata.obs[celltype_l3].isin(list(dictionary_undersample.keys()))].copy()
        # Subset 'adata_temp' using dictionary
        for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype_l3).indices.items():
            lst.append(np.random.choice(cell_type_indices, dictionary_undersample[cell_type], replace=False))
        annotation = np.unique(adata.obs[celltype_l3]).tolist()
        for i in list(dictionary_undersample.keys()):
            if i in annotation:
                annotation.remove(i)
        adata_leave = adata[adata.obs[celltype_l3].isin(annotation)].copy()
    elif celltype_l2:
        adata_temp = adata[adata.obs[celltype_l2].isin(list(dictionary_undersample.keys()))].copy()
        # Subset 'adata_temp' using dictionary
        for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype_l2).indices.items():
            lst.append(np.random.choice(cell_type_indices, dictionary_undersample[cell_type], replace=False))
        annotation = np.unique(adata.obs[celltype_l2]).tolist()
        for i in list(dictionary_undersample.keys()):
            if i in annotation:
                annotation.remove(i)
        adata_leave = adata[adata.obs[celltype_l2].isin(annotation)].copy()
    elif celltype_l1:
        adata_temp = adata[adata.obs[celltype_l1].isin(list(dictionary_undersample.keys()))].copy()
        # Subset 'adata_temp' using dictionary
        for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype_l1).indices.items():
            lst.append(np.random.choice(cell_type_indices, dictionary_undersample[cell_type], replace=False))
        annotation = np.unique(adata.obs[celltype_l1]).tolist()
        for i in list(dictionary_undersample.keys()):
            if i in annotation:
                annotation.remove(i)
        adata_leave = adata[adata.obs[celltype_l1].isin(annotation)].copy()
        
    # Subset 'adata_temp' using randomly selected cells
    adata_temp = adata_temp[np.concatenate(lst)].copy()

    # Subset original 'adata' object using 'adata_leave' and subsetted 'adata_temp' obs_names
    adata_balanced = adata[list(adata_temp.obs_names) + list(adata_leave.obs_names)].copy()
    print(f'Successfully undersampled cell types: {", ".join(str(element) for element in list(dictionary_undersample.keys()))}')     
    print()
    del adata_leave, adata_temp

    # Oversample
    
    # Create dataset for model training
    data = pd.DataFrame(data = adata_balanced.X.toarray(), columns = adata_balanced.var_names)
    
    # Add celltype to data
    if celltype_l5 != None:
        data['celltype_l1'] = adata_balanced.obs[celltype_l1].values
        data['celltype_l2'] = adata_balanced.obs[celltype_l2].values
        data['celltype_l3'] = adata_balanced.obs[celltype_l3].values
        data['celltype_l4'] = adata_balanced.obs[celltype_l4].values
        data['celltype_l5'] = adata_balanced.obs[celltype_l5].values
    elif (celltype_l5 == None) and (celltype_l4 != None):
        data['celltype_l1'] = adata_balanced.obs[celltype_l1].values
        data['celltype_l2'] = adata_balanced.obs[celltype_l2].values
        data['celltype_l3'] = adata_balanced.obs[celltype_l3].values
        data['celltype_l4'] = adata_balanced.obs[celltype_l4].values
    elif (celltype_l4 == None) and (celltype_l3 != None):
        data['celltype_l1'] = adata_balanced.obs[celltype_l1].values
        data['celltype_l2'] = adata_balanced.obs[celltype_l2].values
        data['celltype_l3'] = adata_balanced.obs[celltype_l3].values
    elif (celltype_l3 == None) and (celltype_l2 != None):
        data['celltype_l1'] = adata_balanced.obs[celltype_l1].values
        data['celltype_l2'] = adata_balanced.obs[celltype_l2].values
    elif (celltype_l2 == None) and (celltype_l1 != None):
        data['celltype_l1'] = adata_balanced.obs[celltype_l1].values
    else:
        print('Please, indicate at least one cell annotation starting from celltype_l1')
    
    if sample != None:
        data['sample'] = adata_balanced.obs[sample].values
    
    # Creating a dict file 
    dict_l1 = {}
    c = 0
    for i in np.unique(data['celltype_l1']):
        dict_l1[i] = c
        c += 1
    
    celltype_l1_number = [dict_l1[item] for item in data['celltype_l1']]
    data.insert(1, "classes_l1", celltype_l1_number, True)
    del data['celltype_l1']
    dict_multi = [dict_l1]
    
    if 'celltype_l2' in data:
        dict_l2 = {}
        c = 0
        for i in np.unique(data['celltype_l2']):
            dict_l2[i] = c
            c += 1
    
        celltype_l2_number = [dict_l2[item] for item in data['celltype_l2']]
        data.insert(1, "classes_l2", celltype_l2_number, True)
        del data['celltype_l2']
        dict_multi.append(dict_l2)
    if 'celltype_l3' in data:
        dict_l3 = {}
        c = 0
        for i in np.unique(data['celltype_l3']):
            dict_l3[i] = c
            c += 1
    
        celltype_l3_number = [dict_l3[item] for item in data['celltype_l3']]
        data.insert(1, "classes_l3", celltype_l3_number, True)
        del data['celltype_l3']
        dict_multi.append(dict_l3)
    if 'celltype_l4' in data:
        dict_l4 = {}
        c = 0
        for i in np.unique(data['celltype_l4']):
            dict_l4[i] = c
            c += 1
    
        celltype_l4_number = [dict_l4[item] for item in data['celltype_l4']]
        data.insert(1, "classes_l4", celltype_l4_number, True)
        del data['celltype_l4']
        dict_multi.append(dict_l4)
    if 'celltype_l5' in data:
        dict_l5 = {}
        c = 0
        for i in np.unique(data['celltype_l5']):
            dict_l5[i] = c
            c += 1
    
        celltype_l5_number = [dict_l5[item] for item in data['celltype_l5']]
        data.insert(1, "classes_l5", celltype_l5_number, True)
        del data['celltype_l5']
        dict_multi.append(dict_l5)
    # Add sample
    if 'sample' in data:
        dict_sample = {}
        c = 0
        for i in np.unique(data['sample']):
            dict_sample[i] = c
            c += 1
    
        sample_number = [dict_sample[item] for item in data['sample']]
        data.insert(1, "sample_number", sample_number, True)
        del data['sample']
    
    if 'classes_l5' in data:
        data_y = data['classes_l5'].copy()
    elif 'classes_l4' in data:
        data_y = data['classes_l4'].copy()
    elif 'classes_l3' in data:
        data_y = data['classes_l3'].copy()
    elif 'classes_l2' in data:
        data_y = data['classes_l2'].copy()
    elif 'classes_l1' in data:
        data_y = data['classes_l1'].copy()
    
    # Convert dictionary
    dictionary_number = {}
    if 'classes_l5' in data:
        for i, j in dict_multi[4].items():
            if i in dictionary_oversample.keys():
                dictionary_number[j] = dictionary_oversample[i]
    elif 'classes_l4' in data:
        for i, j in dict_multi[3].items():
            if i in dictionary_oversample.keys():
                dictionary_number[j] = dictionary_oversample[i]
    elif 'classes_l3' in data:
        for i, j in dict_multi[2].items():
            if i in dictionary_oversample.keys():
                dictionary_number[j] = dictionary_oversample[i]
    elif 'classes_l2' in data:
        for i, j in dict_multi[1].items():
            if i in dictionary_oversample.keys():
                dictionary_number[j] = dictionary_oversample[i]
    elif 'classes_l1' in data:
        for i, j in dict_multi[0].items():
            if i in dictionary_oversample.keys():
                dictionary_number[j] = dictionary_oversample[i]
                
    # Oversample selected cel types
    ros = RandomOverSampler(sampling_strategy = dictionary_number, shrinkage = shrinkage)
    data_oversampled, data_y_oversampled = ros.fit_resample(data, data_y)
    del data_y_oversampled, data_y
    
    # Define get_key function for dictionaries
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
    
    # Create meta for adata_oversampled
    meta = []
    
    if 'classes_l1' in data_oversampled:
        meta.append(data_oversampled['classes_l1'].copy())
        del data_oversampled['classes_l1']
    if 'classes_l2' in data_oversampled:
        meta.append(data_oversampled['classes_l2'].copy())
        del data_oversampled['classes_l2']
    if 'classes_l3' in data_oversampled:
        meta.append(data_oversampled['classes_l3'].copy())
        del data_oversampled['classes_l3']
    if 'classes_l4' in data_oversampled:
        meta.append(data_oversampled['classes_l4'].copy())
        del data_oversampled['classes_l4']
    if 'classes_l5' in data_oversampled:
        meta.append(data_oversampled['classes_l5'].copy())
        del data_oversampled['classes_l5']
        
    sample = []
    if 'sample_number' in data_oversampled:
        sample.append(data_oversampled['sample_number'].copy())
        del data_oversampled['sample_number']
    
    # Create adata_oversampled
    adata_balanced = anndata.AnnData(X = sparse.csr_matrix(data_oversampled.values),
                                     var = adata.var)
                                        
    # Add celltype to adata_oversampled
    for i in range(len(dict_multi)):
        annotation_i = [get_key(dict_multi[i], meta_i) for meta_i in meta[i].astype(dtype=int)]
        adata_balanced.obs['celltype_l' + f'{i+1}'] = annotation_i
        
    # Add sample to adata_oversampled
    if dict_sample:
        annotation_i = [get_key(dict_sample, sample_i) for sample_i in sample[0].astype(dtype=int)]
        adata_balanced.obs['sample'] = annotation_i
        print(f'Successfully oversampled cell types: {", ".join(str(element) for element in list(dictionary_oversample.keys()))}')     
        
    return adata_balanced


# Function for oversample selected cell types
def oversample(
    adata,
    dictionary = None,
    sample = None,
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    shrinkage = 1
):
    '''
    Oversample some cell types in AnnData object.
    Returns adata_oversampled with updated matrix and adata_oversampled.obs with given celltypes levels and sample.
    If you give counts function returns counts.
    If you give normalized data function returns normalized data.  
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Function uses adata.X for oversample.
    dictionary : dict, (default: None)
        Dictionary with cell type to be oversampled as key and number of cells you want to get as value.
        Dictionary should only include cell types from the most detailed level of annotation.
    sample : str, (default: None)
        Samples names key in adata.obs dataframe.
    celltype_l1 : str, (default: None)
        First level of cell annotation. Key in adata.obs dataframe.
    celltype_l2 : str, (default: None)
        Second level of cell annotation. Key in adata.obs dataframe.
    celltype_l3 : str, (default: None)
        Third level of cell annotation. Key in adata.obs dataframe.
    celltype_l4 : str, (default: None)
        Forth level of cell annotation. Key in adata.obs dataframe.
    celltype_l5 : str, (default: None)
        Fifth level of cell annotation. Key in adata.obs dataframe.
    shrinkage : float, (default: 1.0)
        Parameter controlling the shrinkage applied to the covariance matrix 
        when a smoothed bootstrap is generated. 
    
    '''
    
    # Create dataset for model training
    data = pd.DataFrame(data=adata.X.toarray(), columns=adata.var_names)
    
    # Add celltype to data
    if celltype_l5 != None:
        data['celltype_l1'] = adata.obs[celltype_l1].values
        data['celltype_l2'] = adata.obs[celltype_l2].values
        data['celltype_l3'] = adata.obs[celltype_l3].values
        data['celltype_l4'] = adata.obs[celltype_l4].values
        data['celltype_l5'] = adata.obs[celltype_l5].values
    elif (celltype_l5 == None) and (celltype_l4 != None):
        data['celltype_l1'] = adata.obs[celltype_l1].values
        data['celltype_l2'] = adata.obs[celltype_l2].values
        data['celltype_l3'] = adata.obs[celltype_l3].values
        data['celltype_l4'] = adata.obs[celltype_l4].values
    elif (celltype_l4 == None) and (celltype_l3 != None):
        data['celltype_l1'] = adata.obs[celltype_l1].values
        data['celltype_l2'] = adata.obs[celltype_l2].values
        data['celltype_l3'] = adata.obs[celltype_l3].values
    elif (celltype_l3 == None) and (celltype_l2 != None):
        data['celltype_l1'] = adata.obs[celltype_l1].values
        data['celltype_l2'] = adata.obs[celltype_l2].values
    elif (celltype_l2 == None) and (celltype_l1 != None):
        data['celltype_l1'] = adata.obs[celltype_l1].values
    else:
        print('Please, indicate at least one cell annotation starting from celltype_l1')
    
    if sample != None:
        data['sample'] = adata.obs[sample].values
    
    # Creating a dict file 
    dict_l1 = {}
    c = 0
    for i in np.unique(data['celltype_l1']):
        dict_l1[i] = c
        c += 1
    
    celltype_l1_number = [dict_l1[item] for item in data['celltype_l1']]
    data.insert(1, "classes_l1", celltype_l1_number, True)
    del data['celltype_l1']
    dict_multi = [dict_l1]
    
    if 'celltype_l2' in data:
        dict_l2 = {}
        c = 0
        for i in np.unique(data['celltype_l2']):
            dict_l2[i] = c
            c += 1
    
        celltype_l2_number = [dict_l2[item] for item in data['celltype_l2']]
        data.insert(1, "classes_l2", celltype_l2_number, True)
        del data['celltype_l2']
        dict_multi.append(dict_l2)
    if 'celltype_l3' in data:
        dict_l3 = {}
        c = 0
        for i in np.unique(data['celltype_l3']):
            dict_l3[i] = c
            c += 1
    
        celltype_l3_number = [dict_l3[item] for item in data['celltype_l3']]
        data.insert(1, "classes_l3", celltype_l3_number, True)
        del data['celltype_l3']
        dict_multi.append(dict_l3)
    if 'celltype_l4' in data:
        dict_l4 = {}
        c = 0
        for i in np.unique(data['celltype_l4']):
            dict_l4[i] = c
            c += 1
    
        celltype_l4_number = [dict_l4[item] for item in data['celltype_l4']]
        data.insert(1, "classes_l4", celltype_l4_number, True)
        del data['celltype_l4']
        dict_multi.append(dict_l4)
    if 'celltype_l5' in data:
        dict_l5 = {}
        c = 0
        for i in np.unique(data['celltype_l5']):
            dict_l5[i] = c
            c += 1
    
        celltype_l5_number = [dict_l5[item] for item in data['celltype_l5']]
        data.insert(1, "classes_l5", celltype_l5_number, True)
        del data['celltype_l5']
        dict_multi.append(dict_l5)
    # Add sample
    if 'sample' in data:
        dict_sample = {}
        c = 0
        for i in np.unique(data['sample']):
            dict_sample[i] = c
            c += 1
    
        sample_number = [dict_sample[item] for item in data['sample']]
        data.insert(1, "sample_number", sample_number, True)
        del data['sample']
    
    if 'classes_l5' in data:
        data_y = data['classes_l5'].copy()
    elif 'classes_l4' in data:
        data_y = data['classes_l4'].copy()
    elif 'classes_l3' in data:
        data_y = data['classes_l3'].copy()
    elif 'classes_l2' in data:
        data_y = data['classes_l2'].copy()
    elif 'classes_l1' in data:
        data_y = data['classes_l1'].copy()
    
    # Convert dictionary
    dictionary_number = {}
    if 'classes_l5' in data:
        for i, j in dict_multi[4].items():
            if i in dictionary.keys():
                dictionary_number[j] = dictionary[i]
    elif 'classes_l4' in data:
        for i, j in dict_multi[3].items():
            if i in dictionary.keys():
                dictionary_number[j] = dictionary[i]
    elif 'classes_l3' in data:
        for i, j in dict_multi[2].items():
            if i in dictionary.keys():
                dictionary_number[j] = dictionary[i]
    elif 'classes_l2' in data:
        for i, j in dict_multi[1].items():
            if i in dictionary.keys():
                dictionary_number[j] = dictionary[i]
    elif 'classes_l1' in data:
        for i, j in dict_multi[0].items():
            if i in dictionary.keys():
                dictionary_number[j] = dictionary[i]
                
    # Oversample selected cel types
    ros = RandomOverSampler(sampling_strategy = dictionary_number, shrinkage = shrinkage)
    data_oversampled, data_y_oversampled = ros.fit_resample(data, data_y)
    del data_y_oversampled, data_y
    
    # Define get_key function for dictionaries
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
    
    # Create meta for adata_oversampled
    meta = []
    
    if 'classes_l1' in data_oversampled:
        meta.append(data_oversampled['classes_l1'].copy())
        del data_oversampled['classes_l1']
    if 'classes_l2' in data_oversampled:
        meta.append(data_oversampled['classes_l2'].copy())
        del data_oversampled['classes_l2']
    if 'classes_l3' in data_oversampled:
        meta.append(data_oversampled['classes_l3'].copy())
        del data_oversampled['classes_l3']
    if 'classes_l4' in data_oversampled:
        meta.append(data_oversampled['classes_l4'].copy())
        del data_oversampled['classes_l4']
    if 'classes_l5' in data_oversampled:
        meta.append(data_oversampled['classes_l5'].copy())
        del data_oversampled['classes_l5']
        
    sample = []
    if 'sample_number' in data_oversampled:
        sample.append(data_oversampled['sample_number'].copy())
        del data_oversampled['sample_number']
    
    # Create adata_oversampled
    adata_oversampled = anndata.AnnData(X = sparse.csr_matrix(data_oversampled.values),
                                        var = adata.var)
                                        
    # Add celltype to adata_oversampled
    for i in range(len(dict_multi)):
        annotation_i = [get_key(dict_multi[i], meta_i) for meta_i in meta[i].astype(dtype=int)]
        adata_oversampled.obs['celltype_l' + f'{i+1}'] = annotation_i
        print(f'Successfully added celltype_l{i+1} to adata_oversampled.obs')

    # Add sample to adata_oversampled
    if dict_sample:
        annotation_i = [get_key(dict_sample, sample_i) for sample_i in sample[0].astype(dtype=int)]
        adata_oversampled.obs['sample'] = annotation_i
        print(f'Successfully added sample to adata_oversampled.obs')     
        
    return adata_oversampled


# Function for undersample specific cell types
def undersample(
    adata,
    dictionary = None,
    celltype = None
):
    '''
    Undersample some cell types in AnnData object.
    Returns subsetted adata_undersampled object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    dictionary : dict, (default: None)
        Dictionary with cell type to be undersampled as key and number of cells you want to get as value.
        Dictionary should only include cell types from the most detailed level of annotation.
    celltype : str, (default: None)
        Level of cell annotation used in dictionary. Key in adata.obs dataframe.        
    
    '''
    # Split 'adata' to 'adata_temp' with cell types to subset and 'adata_leave' with cell types to leave without subset
    lst = []
    adata_temp = adata[adata.obs[celltype].isin(list(dictionary.keys()))].copy()
    # Subset 'adata_temp' using dictionary
    for cell_type, cell_type_indices in adata_temp.obs.groupby(celltype).indices.items():
        lst.append(np.random.choice(cell_type_indices, dictionary[cell_type], replace=False))
    annotation = np.unique(adata.obs[celltype]).tolist()
    for i in list(dictionary.keys()):
        if i in annotation:
            annotation.remove(i)
    adata_leave = adata[adata.obs[celltype].isin(annotation)].copy()
    
    # Subset 'adata_temp' using randomly selected cells
    adata_temp = adata_temp[np.concatenate(lst)].copy()

    # Subset original 'adata' object using 'adata_leave' and subsetted 'adata_temp' obs_names
    adata_undersampled = adata[list(adata_temp.obs_names) + list(adata_leave.obs_names)].copy()

    return adata_undersampled


# Function for creation of full report
def report_classif_full(
    adata,
    celltype = None,
    pred_celltype = None,
    save_report = False,
    report_name = 'report.csv',
    save_path = '',
    ndigits = 4
):
    '''
    Returns metrics (precision, recall (also called sensitivity), specificity, f1-score, geometric mean, and index balanced accuracy of the geometric mean) of predicted cell types.
    You should use it after prediction on annotated test dataset (shows results of validation).
    Helps in understanding model quality.

    adata : AnnData
        Annotated data matrix. Previously annotated test dataset not used for model tuning or training.
    celltype : str, , (default: None)
        Level of cell annotation to show metrics. Key in adata.obs dataframe.
    pred_celltype : str, , (default: None)
        Predicted level of cell annotation. Key in adata.obs dataframe.
    save_report : bool, (default: False)
        Save report as csv file or not.
    report_name : str, (default: 'report.csv')
        Name of a file to save report.
    save_path : path object
        Path to a folder to save report.
    ndigits : int (default: 4)
        Round a number to a given precision in decimal digits.
        
    '''
    
    # Create report with precision, recall/sensitivity, specificity, f1-score, geometric mean, and index balanced accuracy of the geometric mean
    report = metrics.classification_report_imbalanced(adata.obs[celltype].to_numpy(), adata.obs[pred_celltype].to_numpy(), output_dict = True, digits = ndigits)
    report = pd.DataFrame(report)
    del report['avg_pre'], report['avg_rec'], report['avg_spe'], report['avg_f1'], report['avg_geo'], report['avg_iba'], report['total_support']
    report = report.transpose()
    report['sup'] = report['sup'].astype('int')
    
    # Rename columns
    report = report.rename(columns = {'pre': 'precision', 
                                      'rec': 'recall/sensitivity', 
                                      'spe': 'specificity', 
                                      'f1': 'f1-score', 
                                      'geo': 'geometric mean', 
                                      'iba': 'index balanced accuracy', 
                                      'sup': 'number of cells'
                                     })

    # Calculate balanced accuracy and create list wih it
    lst_bal_acc = [round(np.mean(report['recall/sensitivity']), ndigits = ndigits)]
    i = 0
    while i < 6:
        lst_bal_acc.append('')
        i+=1
    
    # Add avg rows
    report.loc['macro avg'] = [round(np.mean(report['precision']), ndigits = ndigits),
                               round(np.mean(report['recall/sensitivity']), ndigits = ndigits),
                               round(np.mean(report['specificity']), ndigits = ndigits),
                               round(np.mean(report['f1-score']), ndigits = ndigits),
                               round(np.mean(report['geometric mean']), ndigits = ndigits),
                               round(np.mean(report['index balanced accuracy']), ndigits = ndigits), '']
    report.loc['weighted avg'] = [round(np.sum(report['precision'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['recall/sensitivity'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['specificity'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['f1-score'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['geometric mean'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['index balanced accuracy'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),'']

    # Round data 
    report['precision'] = round(report['precision'], ndigits = ndigits)
    report['recall/sensitivity'] = round(report['recall/sensitivity'], ndigits = ndigits)
    report['specificity'] = round(report['specificity'], ndigits = ndigits)
    report['f1-score'] = round(report['f1-score'], ndigits = ndigits)
    report['geometric mean'] = round(report['geometric mean'], ndigits = ndigits)
    report['index balanced accuracy'] = round(report['index balanced accuracy'], ndigits = ndigits)

    # Add accuracy
    lst_acc = [round(metrics_.accuracy_score(adata.obs[celltype].to_numpy(), adata.obs[pred_celltype].to_numpy()), ndigits = ndigits)]
    i = 0
    while i < 6:
        lst_acc.append('')
        i+=1
    report.loc['Accuracy'] = lst_acc
    report.loc['Balanced accuracy'] = lst_bal_acc
    del lst_acc, lst_bal_acc
    
    # Save report to .csv
    if save_report:
        report.to_csv(os.path.join(save_path, report_name).replace("\\","/"))
        print('Successfully saved report')
        print()

    return report


# Function to find predition status (correct or incorrect prediction)
def pred_status(
    adata,
    celltype = None,
    pred_celltype = None,
    key_added = 'pred_status'
):
    '''
    Find correct and incorrect predictions.
    Returns prediction status in adata.obs.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Function uses adata.X for oversample.
    celltype :  str, (default: None)
        Cell annotation. Key in adata.obs dataframe.
    pred_celltype :  str, (default: None)
        Predicted cell annotation. Key in adata.obs dataframe.
    key_added : str, (default: 'pred_status')
        Key to add in adata.obs
    
    '''

    adata.obs[key_added] = adata.obs[celltype] == adata.obs[pred_celltype]
    adata.obs[key_added] = adata.obs[key_added].astype('str')
    adata.obs[key_added] = adata.obs[key_added].replace('True', 'correct')
    adata.obs[key_added] = adata.obs[key_added].replace('False', 'incorrect')
    adata.uns[key_added  + '_colors'] = ['#3A3AFF', '#FF3737']


# Function for visualization of cell type prediction using confusion matrix
def conf_matrix(
    adata,
    celltype = None,
    pred_celltype = None,
    fmt = ".2f",
    annot = True,
    cmap = "Blues",
    ndigits_metrics = 3,
    grid = False,
    **kwargs
):
    '''
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Function uses adata.X for oversample.
    celltype :  str, (default: None)
        Cell annotation. Key in adata.obs dataframe.
    pred_celltype :  str, (default: None)
        Predicted cell annotation. Key in adata.obs dataframe.
    fmt : str, optional
        String formatting code to use when adding annotations.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the data. Note that DataFrames will match on position, not index.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    ndigits_metrics : int (default: 3)
        Round a n accuracy and balanced accuracy to a given precision in decimal digits.
    grid : bool (default: False)
        Show or hide grid lines.
    **kwargs: other keyword arguments
        All other keyword arguments are passed to
        sns.heatmap
    '''

    # Create confusion matrix
    cm = metrics_.confusion_matrix(adata.obs[celltype], adata.obs[pred_celltype])

    # Calculate accuracy
    accuracy  = round(np.trace(cm) / float(np.sum(cm)), ndigits = ndigits_metrics)
    accuracy = f"\n\nAccuracy={accuracy}"
    
    # Calculate balanced accuracy
    report = metrics.classification_report_imbalanced(adata.obs[celltype].to_numpy(), adata.obs[pred_celltype].to_numpy(), output_dict = True)
    report = pd.DataFrame(report)
    del report['avg_pre'], report['avg_rec'], report['avg_spe'], report['avg_f1'], report['avg_geo'], report['avg_iba'], report['total_support']
    report = report.transpose()
    bal_accuracy = round(np.mean(report['rec']), ndigits = ndigits_metrics)
    bal_accuracy = f"\n\nBalanced accuracy={bal_accuracy}"
    del report

    # Convert confusion matrix to dataframe
    celltypes = np.unique(adata.obs[celltype])
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    
    plt.grid(grid)
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, **kwargs)
    plt.xlabel('Predicted' + accuracy + bal_accuracy)
    plt.ylabel("Observed")
    

# Function for calculation of sensitivity and specificity of trained model
def report_classif_sens_spec(
    adata,
    celltype = None,
    pred_celltype = None,
    save_report = False,
    report_name = 'report_sens_spec.csv',
    save_path = '',
    ndigits = 3
):
    '''
    Returns specificity and recall (also called sensitivity) metrics of predicted cell types.
    You should use it after prediction on annotated test dataset (shows results of validation).
    Helps in understanding model quality.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Previously annotated test dataset not used for model tuning or training.
    celltype : str, , (default: None)
        Level of cell annotation to show metrics. Key in adata.obs dataframe.
    pred_celltype : str, , (default: None)
        Predicted level of cell annotation. Key in adata.obs dataframe.
    save_report : bool, (default: False)
        Save report as csv file or not.
    report_name : str, (default: 'report_sens_spec.csv')
        Name of a file to save report.
    save_path : path object
        Path to a folder to save report.
    ndigits : int (default: 3)
        Round a number to a given precision in decimal digits.
    '''

    # Create report with sensitivity and specificity
    report = metrics.sensitivity_specificity_support(adata.obs[celltype].to_numpy(), adata.obs[pred_celltype].to_numpy(), )

    # Add cell types names and column names
    report = pd.DataFrame(report, columns=np.unique(adata.obs[celltype]).tolist(), index = ['recall/sensitivity', 'specificity', 'number of cells']).transpose()
    
    # Round data 
    report['recall/sensitivity'] = round(report['recall/sensitivity'], ndigits = ndigits)
    report['specificity'] = round(report['specificity'], ndigits = ndigits)
    report['number of cells'] = report['number of cells'].astype('int')

    # Add avg rows
    report.loc['macro avg'] = [round(np.mean(report['recall/sensitivity']), ndigits = ndigits),
                               round(np.mean(report['specificity']), ndigits = ndigits), '']
    report.loc['weighted avg'] = [round(np.sum(report['recall/sensitivity'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits),
                                  round(np.sum(report['specificity'][:-1] * report['number of cells'])/np.sum(report['number of cells'][:-1]), ndigits = ndigits), '']

    # Save report to .csv
    if save_report:
        report.to_csv(os.path.join(save_path, report_name).replace("\\","/"))
        print('Successfully saved report')
        print()
        
    return report


# Function for calculation of regression metrics of trained model
def report_reg(
    adata_prot,
    adata_pred_prot,
    multioutput = 'uniform_average',
    save_report = False,
    report_name = 'report_regression.csv',
    save_path = '',
    ndigits = 3
):
    '''
    Returns multiple metrics of cell surface proteins prediction.
    Root mean squared error (RMSE), mean absolute error (MeanAE), median absolute error (MedianAE) : lower value - better prediction.
    Coefficient of determination (R² score), explained variance score (EVS) : higher value - better prediction
    
    Parameters
    ----------
    adata_prot : AnnData
        Annotated data matrix with proteins. Test dataset not used for model tuning or training.
    adata_pred_prot : AnnData
        Annotated data matrix with predicted proteins.
    multioutput : {‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,), (default=’uniform_average’)
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        ‘raw_values’ : Returns a full set of errors in case of multioutput input.
        ‘uniform_average’ : Errors of all outputs are averaged with uniform weight.
    save_report : bool, (default: False)
        Save report as csv file or not.
    report_name : str, (default: 'report_sens_spec.csv')
        Name of a file to save report.
    save_path : path object
        Path to a folder to save report.
    ndigits : int (default: 3)
        Round a number to a given precision in decimal digits.
    '''

    # Create DataFrames of predicted and real data
    data_adt = pd.DataFrame(data = adata_prot.X.toarray(), columns = adata_prot.var_names)
    data_pred_adt = pd.DataFrame(data = adata_pred_prot.X.toarray(), columns = adata_pred_prot.var_names)

    # Root mean squared error
    report_RMSE = metrics_.root_mean_squared_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = multioutput)

    # Mean absolute error
    report_MeanAE = metrics_.mean_absolute_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = multioutput)

    # Median absolute error
    report_MedianAE = metrics_.median_absolute_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = multioutput)

    # Explained variance score
    report_EVS = metrics_.explained_variance_score(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = multioutput)
    
    # Coefficient of determination (R² score)
    report_r2_score = metrics_.r2_score(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = multioutput)

    # Create report
    report = pd.DataFrame()
    # Round data 
    report['EVS'] = [round(report_EVS, ndigits=ndigits)]
    report['r2_score'] = [round(report_r2_score, ndigits=ndigits)]
    report['RMSE'] = [round(report_RMSE, ndigits=ndigits)]
    report['MedianAE'] = [round(report_MedianAE, ndigits=ndigits)]
    report['MeanAE'] = [round(report_MeanAE, ndigits=ndigits)]
    
    report.loc['EVS/r2_score'] = ['higher value - better prediction', '', '', '', '']
    report.loc['RMSE/MedianAE/MeanAE'] = ['lower value - better prediction', '', '', '', '']
    report = report.rename(index = {0: "score"})
    
    # Save report to .csv
    if save_report:
        report.to_csv(os.path.join(save_path, report_name).replace("\\","/"))
        print('Successfully saved report')
        print()
        
    return report


# Function for defining regression status
def regres_status(
    adata_prot,
    adata_pred_prot,
    metric = 'RMSE'
):
    '''
    Compute regression status of cells to visualize on UMAP. 

    Parameters
    ----------
    adata_prot : AnnData
        Annotated data matrix with proteins. Test dataset not used for model tuning or training.
    adata_pred_prot : AnnData
        Annotated data matrix with predicted proteins.
    metric : str (default: 'RMSE')
        Metric used for regression status calculation.
        Available metrics: RMSE, MeanAE, MedianAE, EVS, r2_score.
        Root mean squared error (RMSE), mean absolute error (MeanAE), median absolute error (MedianAE) : lower value - better prediction.
        Coefficient of determination (R² score), explained variance score (EVS) : higher value - better prediction
        
    '''

    # Create DataFrames of predicted and real data
    data_adt = pd.DataFrame(data = adata_prot.X.toarray().transpose(), columns = adata_prot.obs_names)
    data_pred_adt = pd.DataFrame(data = adata_pred_prot.X.toarray().transpose(), columns = adata_pred_prot.obs_names)
    
    # Root mean squared error
    if metric == 'RMSE':
        report = metrics_.root_mean_squared_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = 'raw_values')
    # Mean absolute error
    elif metric == 'MeanAE':
        report = metrics_.mean_absolute_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = 'raw_values')
    # Median absolute error
    elif metric == 'MedianAE':
        report = metrics_.median_absolute_error(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = 'raw_values')
    # Explained variance score
    elif metric == 'EVS':
        report = metrics_.explained_variance_score(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = 'raw_values')
    # Coefficient of determination (R² score)
    elif metric == 'r2_score':
        report = metrics_.r2_score(y_true = data_adt.values, y_pred = data_pred_adt.values, multioutput = 'raw_values')
    
    del data_adt, data_pred_adt

    # Add metric to predicted adata
    adata_pred_prot.obs['regres_status_' + metric] = report.copy()
    del report
    

# Function for calculation Pearson correlation coefficient per protein
def pearson_coef_prot(
    adata_prot,
    adata_pred_prot,
    protein,
    protein_pred,
    ndigits = 3,
    print_res = False
):
    '''
    Compute Pearson correlation coefficient of predicted protein. 
    Varies between -1 and +1. 

    Parameters
    ----------
    adata_prot : AnnData
        Annotated data matrix with proteins. Test dataset not used for model tuning or training.
    adata_pred_prot : AnnData
        Annotated data matrix with predicted proteins.
    protein : str
        Name of the protein in adata_prot.
    protein_pred : str
        Name of the predicted protein in adata_pred_prot.
    ndigits : int, (default: 3)
        Round a number to a given precision in decimal digits.
    print_res : bool, (default: False)
        Print results or not.
        
    Returns dictionary with Pearson correlation coefficient and p-value.
    Values close to 1 indicate strong positive correlation, and values close to -1 indicate strong negative correlation.
        
    '''

    # Calculate Pearson correlation coefficient 
    res = stats.pearsonr(adata_prot[:, protein].X.T.toarray()[0], 
                         adata_pred_prot[:, protein_pred].X.T.toarray()[0]
                        )
    
    # Create results dictionary
    results = {'Pearson coefficient' : round(res.correlation, ndigits = ndigits),
               'p-value' : "{:.3e}".format(res.pvalue)}
    if print_res == True:
        print(f'Pearson coefficient = {round(res.correlation, ndigits = ndigits)}, p-value = {"{:.3e}".format(res.pvalue)}')
        
    return results


# Function for calculation Spearman correlation coefficient per protein
def spearman_coef_prot(
    adata_prot,
    adata_pred_prot,
    protein,
    protein_pred,
    ndigits = 3,
    print_res = False
):
    '''
    Compute Spearman correlation coefficient of predicted protein. 
    Varies between -1 and +1. 

    Parameters
    ----------
    adata_prot : AnnData
        Annotated data matrix with proteins. Test dataset not used for model tuning or training.
    adata_pred_prot : AnnData
        Annotated data matrix with predicted proteins.
    protein : str
        Name of the protein in adata_prot.
    protein_pred : str
        Name of the predicted protein in adata_pred_prot.
    ndigits : int, (default: 3)
        Round a number to a given precision in decimal digits.
    print_res : bool, (default: False)
        Print results or not.
        
    Returns dictionary with Spearman correlation coefficient and p-value.
    Values close to 1 indicate strong positive correlation, and values close to -1 indicate strong negative correlation.
        
    '''

    # Calculate Spearman correlation coefficient 
    res = stats.spearmanr(adata_prot[:, protein].X.T.toarray()[0], 
                         adata_pred_prot[:, protein_pred].X.T.toarray()[0]
                        )
    
    # Create results dictionary
    results = {'Spearman coefficient' : round(res.correlation, ndigits = ndigits),
               'p-value' : "{:.3e}".format(res.pvalue)}
    
    if print_res == True:
        print(f'Spearman coefficient = {round(res.correlation, ndigits = ndigits)}, p-value = {"{:.3e}".format(res.pvalue)}')
        
    return results


# Function for calculation Spearman correlation coefficient per protein
def kendalltau_coef_prot(
    adata_prot,
    adata_pred_prot,
    protein,
    protein_pred,
    ndigits = 3,
    print_res = False
):
    '''
    Compute Kendall’s tau, a correlation measure of predicted protein. 
    Varies between -1 and +1. 

    Parameters
    ----------
    adata_prot : AnnData
        Annotated data matrix with proteins. Test dataset not used for model tuning or training.
    adata_pred_prot : AnnData
        Annotated data matrix with predicted proteins.
    protein : str
        Name of the protein in adata_prot.
    protein_pred : str
        Name of the predicted protein in adata_pred_prot.
    ndigits : int, (default: 3)
        Round a number to a given precision in decimal digits.
    print_res : bool, (default: False)
        Print results or not.
        
    Returns dictionary with Kendall’s tau coefficient and p-value.
    Values close to 1 indicate strong agreement, and values close to -1 indicate strong disagreement
        
    '''

    # Calculate Kendall’s tau coefficient 
    res = stats.kendalltau(adata_prot[:, protein].X.T.toarray()[0], 
                           adata_pred_prot[:, protein_pred].X.T.toarray()[0]
                          )
    
    # Create results dictionary
    results = {'kendall Tau' : round(res.correlation, ndigits = ndigits),
               'p-value' : "{:.3e}".format(res.pvalue)}
    
    if print_res == True:
        print(f'Kendall Tau = {round(res.correlation, ndigits = ndigits)}, p-value = {"{:.3e}".format(res.pvalue)}')
        
    return results
    

# Function for count cell types in samples of integrated dataset
def cell_counter(
    adata,
    sample = None,
    celltype = None
):
    '''
    Count cell types in samples. Usefull for integrated/concatenated dataset.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sample : str, (default: None)
        Samples names key in adata.obs dataframe.
    celltype : str, (default: None)
        Level of cell annotation to show metrics. Key in adata.obs dataframe.
    '''
    # Create dataframe to store samples cell types 
    df = pd.DataFrame()
    
    # Add samples to dataframe
    for i in list(np.unique(adata.obs[sample])):
        adata_temp = adata[adata.obs[sample] == i]
        df_temp = pd.DataFrame(adata_temp.obs[celltype].value_counts()).rename(columns={'count' : i})
        df = pd.concat([df, df_temp[i]], axis  = 1, join='outer')
    del adata_temp, df_temp
    
    return df


# Function to get explanations 
def explain(
    adata, 
    celltype = None,
    path_model = '',
    num_cells = 100,
    random_state = 0,
    max_evals = 2000
):
    '''
    Identify the genes that are most important for determining cell type using a model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    celltype : str, (default: None)
        Specific cell type in annotation to calculate explanations.
    path_model : str, path object
        Path to the folder containing the trained scAdam model.
    num_cells : int, (default: 100)
        Number of cells to make explanations. 
        Increasing the number of cells will lead to an increase in computation time.
    random_state : int, (default: 0)
        Controls the random selection of cells from dataset and explainer for reproducibility.
        Pass an int for reproducible output across multiple function calls.
    max_evals : int, (default: 2000)
        The max_evals parameter in SHAP is a tunable setting that significantly affects both 
        the accuracy and computational efficiency of SHAP value calculations. 
        By adjusting this parameter, you can balance between obtaining detailed explanations 
        and managing computational resources effectively. 
        For a larger number of genes, it is necessary to increase max_evals.

    returns explanations of specific cell type.
    
    '''
    
    # load genes of trained model
    features = pd.read_csv(os.path.join(path_model, 'genes.csv').replace("\\","/"))    
    features = list(features['feature_name'])
    print('Successfully loaded list of genes used for training model')
    
    # Create dataset for prediction
    data_genes = adata.raw.var_names.tolist()
    data_predict = pd.DataFrame(adata.raw.X.toarray(), columns = data_genes)
    sorted_dataset = pd.DataFrame(index = [i for i in range(0, len(adata.obs_names))])
    for column in features:
        if column in data_genes:
            sorted_dataset[column] = data_predict[column]
        else:
            sorted_dataset[column] = 0

    # Load dictionary of trained cell types
    with open(os.path.join(path_model, 'dict.txt')) as dict:
        dict = dict.read()
        dict_multi = json.loads(dict)
    print('Successfully loaded dictionary of dataset annotations')

    # Load pretrained model
    loaded_model = TabNetMultiTaskClassifier()
    for file in os.listdir(path_model):
        if file.endswith('.zip'):
            loaded_model.load_model(os.path.join(path_model, file).replace("\\","/"))
            print('Successfully loaded model')

    # Predict cell types
    predictions = loaded_model.predict(sorted_dataset.values)

    # Define get_key function for dictionaries
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
            
    # Add predictions and probabilities to adata
    for i in range(len(dict_multi)):
        prediction_i = [get_key(dict_multi[i], prediction) for prediction in predictions[i].astype(dtype=int)]
        adata.obs['pred_celltype_l' + f'{i+1}'] = prediction_i

    # Select cell type from a predictions
    if celltype in dict_multi[0].keys():
        adata_cell_type = adata[adata.obs['pred_celltype_l1'] == celltype].copy()
        print(f'Cell type "{celltype}" was successfully selected from pred_celltype_l1')
    elif celltype in dict_multi[1].keys():
        adata_cell_type = adata[adata.obs['pred_celltype_l2'] == celltype].copy()
        print(f'Cell type "{celltype}" was successfully selected from pred_celltype_l2')
    elif celltype in dict_multi[2].keys():
        adata_cell_type = adata[adata.obs['pred_celltype_l3'] == celltype].copy()
        print(f'Cell type "{celltype}" was successfully selected from pred_celltype_l3')
    elif celltype in dict_multi[3].keys():
        adata_cell_type = adata[adata.obs['pred_celltype_l4'] == celltype].copy()
        print(f'Cell type "{celltype}" was successfully selected from pred_celltype_l4')
    elif celltype in dict_multi[4].keys():
        adata_cell_type = adata[adata.obs['pred_celltype_l5'] == celltype].copy()
        print(f'Cell type "{celltype}" was successfully selected from pred_celltype_l5')
    else:
        raise ValueError('Wrong name of cell type! There is no such cell type name in any prediction level.')
    
    data_predict = pd.DataFrame(adata_cell_type.raw.X.toarray(), columns = data_genes)
    sorted_dataset = pd.DataFrame(index = [i for i in range(0, len(adata_cell_type.obs_names))])
    for column in features:
        if column in data_genes:
            sorted_dataset[column] = data_predict[column]
        else:
            sorted_dataset[column] = 0
            
    # Get {num_cells} from a sorted_dataset of specific cell type using {random_state}
    sorted_dataset = sorted_dataset.sample(n = num_cells, axis = 0, random_state = random_state)

    # Create explainer function for loaded model
    def explainer_model (
        sorted_dataset,
        loaded_model = loaded_model
    ):
        arr = loaded_model.predict(sorted_dataset.values)
        if celltype in dict_multi[0].keys():
            arr = arr[0].astype('float64') 
        elif celltype in dict_multi[1].keys():
            arr = arr[1].astype('float64') 
        elif celltype in dict_multi[2].keys():
            arr = arr[2].astype('float64') 
        elif celltype in dict_multi[3].keys():
            arr = arr[3].astype('float64') 
        elif celltype in dict_multi[4].keys():
            arr = arr[4].astype('float64') 
        return arr

    # Create explainer using explainer function and sorted_dataset 
    explainer = shap.Explainer(explainer_model, sorted_dataset, seed = random_state)
    # Get explanations
    explanations = explainer(sorted_dataset, max_evals = max_evals)

    print(f'The explanations for "{celltype}" have been completed')

    return explanations


# Function to get gene importance dataframe from explanations
def feature_importance (
    explanations,
    path_model = ''
):

    '''
    Get dataframe with gene importances for specific cell type.
    
    Parameters
    ----------
    explanations : explanations
        Output of function explain().
    path_model : str, path object
        Path to the folder containing the trained scAdam model.

    returns dataframe of gene importances for prediction of specfic cell type.
    
    '''
    # load genes of trained model
    features = pd.read_csv(os.path.join(path_model, 'genes.csv').replace("\\","/"))    
    features = list(features['feature_name'])
    
    # Get DataFrame with gene importances for specific cell type
    shap_values = explanations.values
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(features, vals)), columns=['gene_name', 'gene_importance'])
    feature_importance.sort_values(by = ['gene_importance'], ascending=False, inplace=True)

    return feature_importance


# Function to get fraction of AnnData
def get_frac(
    adata = None, 
    path = None,
    path_save = '',
    celltype = None, 
    fraction = 0.1,
    shuffle = True,
    random_state = 0
):

    '''
    Get fraction of anndata object. Specify path OR anndata object. 
    The function returns a portion of the AnnData object while maintaining the ratio of cell types.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to the AnnData object if AnnData is not loaded into RAM.
    path_save : str, path object
        Path to save fraction of AnnData
    celltype : str, (default: None)
        Cell type annotation. Key in adata.obs dataframe.
    fraction : float or int, (default: 0.1)
        Should be between 0.0 and 1.0 and represent the proportion 
        of the dataset to include in the adata. 
    shuffle : bool, (default: True)
        Whether or not to shuffle the data before subsetting. 
        If shuffle = False, then `celltype` is not used to maintain the same ratio.
    random_state : int, (default: 0)
        Controls the data shuffling and splitting.
        Pass an int for reproducible output across multiple function calls.

    Returns a fraction of the AnnData object while maintaining the same ratio of cell types. 
    This fraction of the AnnData object is also saved as `adata_fraction.h5ad`.
    
    '''

    
    # Get meta data from AnnData
    if path != None:
        adata = sc.read_h5ad(path, backed = 'r+')
        obs = adata.obs
    elif adata is not None:
        obs = adata.obs

    # Subset meta data
    if shuffle:
        _, obs = train_test_split(obs,
                                  test_size = fraction,
                                  stratify = obs[celltype],
                                  shuffle = shuffle,
                                  random_state = random_state)
        del _
    else:
        _, obs = train_test_split(obs,
                                  test_size = fraction,
                                  stratify = None,
                                  shuffle = shuffle,
                                  random_state = random_state)
        del _

    # Get and save adata_fraction
    if adata.isbacked:
        adata_fraction = adata[obs.index].copy(os.path.join(path_save, 'adata_fraction.h5ad'))
        adata_fraction = sc.read_h5ad(os.path.join(path_save, 'adata_fraction.h5ad'))
    else:
        adata_fraction = adata[obs.index].copy()
        adata_fraction.write_h5ad(os.path.join(path_save, 'adata_fraction.h5ad'))
    return adata_fraction


# Function to get samples from AnnData
def get_samples(
    adata = None, 
    path = None,
    path_save = '',
    sample_col = None, 
    samples = None
):

    '''
    Get samples from AnnData object. Specify path OR AnnData object. 
    The function returns a new AnnData with selected samples.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to the AnnData object if AnnData is not loaded into RAM.
    path_save : str, path object
        Path to save fraction of AnnData
    sample_col : str, (default: None)
        Key in adata.obs dataframe with samples names.
    samples : list, (default: None)
        List of samples names in adata.obs[sample_col].

    Returns a samples from the AnnData object.
    AnnData with samples is also saved as `adata_samples.h5ad`.
    
    '''

    # Get meta data from AnnData
    if path != None:
        adata = sc.read_h5ad(path, backed = 'r+')

    # Get and save adata_samples
    if adata.isbacked:
        adata_samples = adata[adata.obs[sample_col].isin(samples)].copy(os.path.join(path_save, 'adata_samples.h5ad'))
        adata_samples = sc.read_h5ad(os.path.join(path_save, 'adata_samples.h5ad'))
    else:
        adata_samples = adata[adata.obs[sample_col].isin(samples)].copy()
        adata_samples.write_h5ad(os.path.join(path_save, 'adata_samples.h5ad'))
    return adata_samples


# Function to find difference between clusters based on a specific scores
def clust_diff(
    adata,
    groupby = None,
    group1 = None,
    group2 = None,
    score1 = None,
    score2 = None,
    plot = True,
    thresh = 0.02,
    fill = True,
    alpha = 0.5,
    **kwargs
):
    '''
    Calculates metrics to Integral of absolute density difference and Mutual Information between two clusterings.
    Each metric follows the principle that the higher the value, the better the clusters separate the selected scores.
    
    adata : AnnData or MuData
        Annotated data or Multimodal data.
    groupby : str
        The key of the grouping in AnnData.obs or MuData.obs.
    group1 : str
        Cluster in groupby.
    group2 : str
        Cluster in groupby.
    score1 : str
        Score 1 in AnnData.obs or MuData.obs calculated using scanpy.tl.score_genes.
    score2 : str
        Score 2 in AnnData.obs or MuData.obs calculated using scanpy.tl.score_genes.
    plot : bool, (default: True)
        Show kernel density estimate plot or not.
    thresh : float in [0, 1], (default: 0.02)
        Lowest iso-proportion level at which to draw a contour line.
    fill : bool or None, (default: True)
        If True, fill in the area under univariate density curves or between bivariate contours.
    alpha : float or None, (default: 0.5)
        Transparency of the rectangle and connector lines.
    kwargs
        Other keyword arguments are passed to seaborn.kdeplot.

    Returns a plot and Integral of absolute density difference and Mutual Information between two clusterings.
    '''

    # Create DataFrame with groups from groupby 
    df = adata.obs[adata.obs[groupby].isin([group1, group2])][[groupby, score1, score2]]
    df[groupby] = df[groupby].astype(object)
    groups = df.groupby(groupby) 
    group1, group2 = list(groups)[:2]

    # Overlap metric
    def compute_overlap_metric(x1, y1, x2, y2, grid_size=500):
        xmin = min(x1.min(), x2.min())
        xmax = max(x1.max(), x2.max())
        ymin = min(y1.min(), y2.min())
        ymax = max(y1.max(), y2.max())
    
        X, Y = np.mgrid[xmin:xmax:grid_size*1j, ymin:ymax:grid_size*1j]
        positions = np.vstack([X.ravel(), Y.ravel()])
    
        #KDE
        values1 = np.vstack([x1, y1])
        kernel1 = stats.gaussian_kde(values1)
        Z1 = np.reshape(kernel1(positions).T, X.shape)
    
        values2 = np.vstack([x2, y2])
        kernel2 = stats.gaussian_kde(values2)
        Z2 = np.reshape(kernel2(positions).T, X.shape)
    
        # Absolute difference
        diff = np.abs(Z1 - Z2)
        overlap_metric = np.trapz(np.trapz(diff, axis=1), axis=0)
    
        return overlap_metric

    # Mutual information score
    def compute_mutual_information(x, y, bins=20):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = metrics_.mutual_info_score(None, None, contingency=c_xy)
        return mi
        
    # Score1 and score2
    x1, y1 = group1[1][[score1, score2]].values.T
    x2, y2 = group2[1][[score1, score2]].values.T
    metric = compute_overlap_metric(x1, y1, x2, y2)
    mi = compute_mutual_information(df[score1], df[score2])

    if plot:
        plt.title(f"KDE Plot of {score1} vs {score2} \n"
              f"Integral of absolute density difference: {metric:.1f}, \nMutual Information: {mi:.3f}", fontsize=12)
        plt.xlabel(score1, fontsize=10)
        plt.ylabel(score2, fontsize=10)
        
        sns.kdeplot(
            data=df,
            x=score1,
            y=score2,
            hue=groupby,
            thresh=thresh,
            fill=fill,
            alpha=alpha,
            **kwargs
        )
    else:
        print(f"Integral of absolute density difference: {metric:.1f}, \nMutual Information: {mi:.3f}")

    return metric, mi 