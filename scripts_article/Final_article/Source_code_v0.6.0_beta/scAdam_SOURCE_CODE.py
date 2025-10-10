import os 
import pandas as pd
import numpy as np
import torch 
import json 
import fsspec
import optuna
import functools
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Function for training model
def train(
    adata, 
    path = '',
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    model_name = 'model_annotation',
    accelerator = 'auto',
    random_state = 0,
    test_size = 0.1,
    n_d = 8, 
    n_a = 8, 
    n_steps = 3, 
    n_shared = 2, 
    cat_emb_dim = 1, 
    n_independent = 2,
    gamma = 1.3,
    momentum = 0.02,
    lr = 0.02, 
    lambda_sparse = 0.001,
    patience = 10, 
    max_epochs = 100,
    batch_size = 1024,
    virtual_batch_size = 128,
    mask_type = 'entmax', 
    eval_metric = ['accuracy'],
    optimizer_fn = torch.optim.AdamW,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.CrossEntropyLoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    verbose = True,
    drop_last = True,
    return_model = False,
    from_unsupervised = True,
    pretraining_ratio = 0.5
):

    '''
    Train custom scAdam model using annotated data matrix (adata).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to create a folder with model, training history, dictionary of cell annotations and genes used for training.
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
    model_name : str, (default: 'model_annotation')
        Name of a folder to save model.
    accelerator : str, (default: 'auto')
        Type of accelerator to use in training model ('cpu', 'cuda'). Set 'auto' for automatic selection.
    random_state : int, (default: 0)
        Controls the data shuffling, splitting to folds and model training.
        Pass an int for reproducible output across multiple function calls.
    test_size : float or int, (default: 0.1)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test cells.
    n_d : int, (default: 8)
        Width of the decision prediction layer. 
        Bigger values gives more capacity to the model with the risk of overfitting. 
        Values typically range from 8 to 64.
    n_a : int, (default: 8) 
        Width of the attention embedding for each mask.
        Values typically range from 8 to 64.
    n_steps : int, (default: 3) 
        Number of steps in the architecture.
        Values typically range from 3 to 10.
    n_shared : int, (default: 2) 
        Number of shared Gated Linear Units at each step.
        Values typically range from 1 to 5.
    cat_emb_dim : int, (default: 1)
        List of embeddings size for each categorical features.
        Values typically range from 1 to 5.
    n_independent : int, (default: 2)
        Number of independent Gated Linear Units layers at each step. 
        Values typically range from 1 to 5.
    gamma : float, (default: 1.3)
        This is the coefficient for feature reusage in the masks. 
        A value close to 1 will make mask selection least correlated between layers. 
        Values typically range from 1.0 to 2.0.
    momentum : float, (default: 0.02)
        Momentum for batch normalization.
        Values typically range from 0.01 to 0.4.
    lr : float, (default: 0.02) 
        Determines the step size at each iteration while moving toward a minimum of a loss function. 
        A large initial learning rate of 0.02  with decay is a good option
    lambda_sparse : float, (default: 0.001) 
        This is the extra sparsity loss coefficient. 
        The bigger this coefficient is, the sparser your model will be in terms of feature selection. 
        Depending on the difficulty of your problem, reducing this value could help.
    patience : int, (default: 10)
        Number of consecutive epochs without improvement before performing early stopping.
        If patience is set to 0, then no early stopping will be performed.
        Note that if patience is enabled, then best weights from best epoch will automatically be loaded at the end of the training.
    max_epochs : int, (default: 100)
        Maximum number of epochs for training.
    batch_size : int, (default: 1024)
        Number of examples per batch. 
        It is highly recomended to tune this parameter.
    virtual_batch_size : int, (default: 128)
        Size of the mini batches used for "Ghost Batch Normalization".
        'virtual_batch_size' should divide 'batch_size'.
    mask_type : str, (default: 'entmax')
        Either "sparsemax" or "entmax". This is the masking function to use for selecting features.
    eval_metric : list, (default: ['accuracy'])
        List of evaluation metrics ('accuracy', 'balanced_accuracy', 'logloss').
        The last metric is used as the target and for early stopping.
    optimizer_fn : func, (default: torch.optim.AdamW)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.CrossEntropyLoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    return_model : bool, (default: False)
        Return model after training or not.
    from_unsupervised : bool, (default: True)
        Use a previously self supervised model as starting weights. 
        Supervised model training included in function.
    pretraining_ratio : float, (default: 0.5)
        Between 0 and 1, percentage of feature to mask for reconstruction.
        Used for supervised model training.
    '''

    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))
    
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

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     
     
    # Save gene names for future prediction
    cols = data.columns
    if celltype_l5 != None:
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4', 'celltype_l5']
    elif (celltype_l5 == None) and (celltype_l4 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4']
    elif (celltype_l4 == None) and (celltype_l3 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3']
    elif (celltype_l3 == None) and (celltype_l2 != None):
        unused = ['celltype_l1', 'celltype_l2']
    elif (celltype_l2 == None) and (celltype_l1 != None):
        unused = ['celltype_l1']
    features = [col for col in cols if col not in unused]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()
    
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
    
    # write a dictionary to model folder
    with open(os.path.join(path, model_name, 'dict.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(dict_multi))
    del dict_multi
    print('Successfully saved dictionary of dataset annotations')
    print()
    
    # Split data for training
    ## Split using 'celltype_l5' if it is given
    if celltype_l5 != None:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l5'])
        del data
    ## Split using 'celltype_l4' if 'celltype_l5' is not given
    elif (celltype_l5 == None) and (celltype_l4 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l4'])
        del data
    ## Split using 'celltype_l3' if 'celltype_l4' is not given
    elif (celltype_l4 == None) and (celltype_l3 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l3'])
        del data
    ## Split using 'celltype_l2' if 'celltype_l3' is not given
    elif (celltype_l3 == None) and (celltype_l2 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l2'])
        del data
    ## Split using 'celltype_l1' if 'celltype_l2' is not given
    elif (celltype_l2 == None) and (celltype_l1 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l1'])
        del data

    print(f'Train dataset contains: {len(train)} cells, it is {round(100*(len(train)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print(f'Test dataset contains: {len(test)} cells, it is {round(100*(len(test)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print()
    
    # Set target list
    target = ['classes_l1']
    if 'classes_l2' in train:
        target.append('classes_l2')
    if 'classes_l3' in train:
        target.append('classes_l3')
    if 'classes_l4' in train:
        target.append('classes_l4')
    if 'classes_l5' in train:
        target.append('classes_l5')
        
    # Variables to store history, explainability and total best score
    cash = {}
    explains = {}
    
    # Create parameters for learning model
    params = {'n_d': n_d, 
              'n_a': n_a, 
              'n_steps': n_steps, 
              'n_shared': n_shared, 
              'cat_emb_dim': cat_emb_dim, 
              'n_independent': n_independent,
              'gamma': gamma,
              'momentum': momentum,
              'optimizer_params': {'lr': lr}, 
              'mask_type': mask_type, 
              'lambda_sparse': lambda_sparse
              }
    # Define accelerator
    if accelerator == 'auto':
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    params["device_name"] = accelerator
    print(f'Accelerator: {accelerator}')
        
    # Get the training features and labels
    train_target = train[target].values
    train_matrix = train[features].values
    del train
    
    # Get the validation features and labels
    test_target = test[target].values
    test_matrix = test[features].values
    del test

    # Augmentation
    aug = ClassificationSMOTE(seed = random_state)
    
    # Create and train unsupervised model
    if from_unsupervised:
        unsupervised_model = TabNetPretrainer(
            **params,
            optimizer_fn = optimizer_fn,
            scheduler_fn = scheduler_fn, 
            scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
            verbose = verbose,
            seed = random_state
        )

        print('Training unsupervised model')
        
        unsupervised_model.fit(
            X_train=train_matrix,
            eval_set=[test_matrix],
            pretraining_ratio=pretraining_ratio,
            #loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last
        )
        
    # Create model
    clf = TabNetMultiTaskClassifier(
        **params,
        optimizer_fn = optimizer_fn,
        scheduler_fn = scheduler_fn, 
        scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
        verbose = verbose,
        seed = random_state
    )
        
    # Train model
    print()
    print('Training supervised model')
    if from_unsupervised:
        clf.fit(
            X_train = train_matrix,
            y_train = train_target,
            eval_set = [(train_matrix, train_target), (test_matrix, test_target)],
            eval_name = ["train", "valid"],
            eval_metric = eval_metric,
            loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last,
            augmentations = aug,
            from_unsupervised=unsupervised_model
        )
    else:
        clf.fit(
            X_train = train_matrix,
            y_train = train_target,
            eval_set = [(train_matrix, train_target), (test_matrix, test_target)],
            eval_name = ["train", "valid"],
            eval_metric = eval_metric,
            loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last,
            augmentations = aug
        )
        
    # Save history and parameters
    # History
    cash = clf.history.history
    cash = pd.DataFrame(cash)
    cash['epoch'] = cash.index
    cash = cash.set_index('epoch')
    cash.to_csv(os.path.join(path, model_name, 'history.csv').replace("\\","/"))
    
    # Parameters
    params["scheduler_params"] = {"step_size": step_size, "gamma": gamma_scheduler}
    params["batch_size"] = batch_size
    params["virtual_batch_size"] = virtual_batch_size
    params["patience"] = patience
    params["max_epochs"] = max_epochs
    with open(os.path.join(path, model_name, 'params.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(params))
        
    print()
    print('Successfully saved training history and parameters')
    
    # Save tabnet model
    clf.save_model(os.path.join(path, model_name, 'model').replace("\\","/"))

    if return_model == True:
        return clf
    
# Function for fine-tuning pretrained model
def warm_start(
    adata, 
    path = '',
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    model_name = 'model_annotation',
    accelerator = 'auto',
    random_state = 0,
    test_size = 0.1,
    n_d = None, 
    n_a = None, 
    n_steps = None, 
    n_shared = None, 
    cat_emb_dim = None, 
    n_independent = None,
    gamma = None,
    momentum = None,
    lr = None, 
    lambda_sparse = None,
    patience = None, 
    max_epochs = None,
    batch_size = None,
    virtual_batch_size = None,
    mask_type = None, 
    eval_metric = ['accuracy'],
    optimizer_fn = torch.optim.AdamW,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.CrossEntropyLoss(),
    step_size = None, 
    gamma_scheduler = None,
    verbose = True,
    drop_last = True,
    return_model = False
):

    '''
    Warm-start training of scAdam model. 
    Warm-start training is a technique in machine learning that involves initializing a model with parameters or states learned from a previously trained model.
    You can use parameters from pretrained model. Also, you can change any of parameters and it will be saved.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to create a folder with model, training history, dictionary of cell annotations and genes used for training.
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
    model_name : str, (default: 'model_annotation')
        Name of a folder to save model.
    accelerator : str, (default: 'auto')
        Type of accelerator to use in training model ('cpu', 'cuda'). Set 'auto' for automatic selection.
    random_state : int, (default: 0)
        Controls the data shuffling, splitting to folds and model training.
        Pass an int for reproducible output across multiple function calls.
    test_size : float or int, (default: 0.1)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test cells.
    n_d : int, (default: None)
        Width of the decision prediction layer. 
        Bigger values gives more capacity to the model with the risk of overfitting. 
        Values typically range from 8 to 64.
    n_a : int, (default: None) 
        Width of the attention embedding for each mask.
        Values typically range from 8 to 64.
    n_steps : int, (default: None) 
        Number of steps in the architecture.
        Values typically range from 3 to 10.
    n_shared : int, (default: None) 
        Number of shared Gated Linear Units at each step.
        Values typically range from 1 to 5.
    cat_emb_dim : int, (default: None)
        List of embeddings size for each categorical features.
        Values typically range from 1 to 5.
    n_independent : int, (default: None)
        Number of independent Gated Linear Units layers at each step. 
        Values typically range from 1 to 5.
    gamma : float, (default: None)
        This is the coefficient for feature reusage in the masks. 
        A value close to 1 will make mask selection least correlated between layers. 
        Values typically range from 1.0 to 2.0.
    momentum : float, (default: None)
        Momentum for batch normalization.
        Values typically range from 0.01 to 0.4.
    lr : float, (default: None) 
        Determines the step size at each iteration while moving toward a minimum of a loss function. 
        A large initial learning rate of 0.02  with decay is a good option
    lambda_sparse : float, (default: None) 
        This is the extra sparsity loss coefficient. 
        The bigger this coefficient is, the sparser your model will be in terms of feature selection. 
        Depending on the difficulty of your problem, reducing this value could help.
    patience : int, (default: 10)
        Number of consecutive epochs without improvement before performing early stopping.
        If patience is set to 0, then no early stopping will be performed.
        Note that if patience is enabled, then best weights from best epoch will automatically be loaded at the end of the training.
    max_epochs : int, (default: 100)
        Maximum number of epochs for training.
    batch_size : int, (default: None)
        Number of examples per batch. 
        It is highly recomended to tune this parameter.
    virtual_batch_size : int, (default: None)
        Size of the mini batches used for "Ghost Batch Normalization".
        'virtual_batch_size' should divide 'batch_size'.
    mask_type : str, (default: None)
        Either "sparsemax" or "entmax". This is the masking function to use for selecting features.
    eval_metric : list, (default: ['accuracy'])
        List of evaluation metrics ('accuracy', 'balanced_accuracy', 'logloss').
        The last metric is used as the target and for early stopping.
    optimizer_fn : func, (default: torch.optim.AdamW)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.CrossEntropyLoss)
        Loss function for training.
    step_size : int, (default: None)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: None) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    return_model : bool, (default: False)
        Return model after training or not.
        
    '''

    if model_name == '':
        print('Define model for fine tuning')
        
    # load genes of pretrained model
    features = pd.read_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"))    
    features = list(features['feature_name'])
    print('Successfully loaded list of genes used for training model')
    print()
    
    # Create dataset for fine tuning
    data_genes = adata.raw.var_names.tolist()
    data_predict = pd.DataFrame(adata.raw.X.toarray(), columns = data_genes)
    data = pd.DataFrame(index = [i for i in range(0, len(adata.obs_names))])
    for column in features:
        if column in data_genes:
            data[column] = data_predict[column]
        else:
            data[column] = 0
            warnings.warn("If the gene is not present in the AnnData object, it will be assigned a value of 0 for all cells.")

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

    # Load dictionary of trained cell types
    with open(os.path.join(path, model_name, 'dict.txt').replace("\\","/")) as dict:
        dict = dict.read()
        dict_multi = json.loads(dict)
    print('Successfully loaded dictionary of dataset annotations')
    print()
    
    # Creating a dict file 
    dict_l1 = dict_multi[0]
    celltype_l1_number = [dict_l1[item] for item in data['celltype_l1']]
    data.insert(1, "classes_l1", celltype_l1_number, True)
    del data['celltype_l1']

    if 'celltype_l2' in data:
        dict_l2 = dict_multi[1]
        celltype_l2_number = [dict_l2[item] for item in data['celltype_l2']]
        data.insert(1, "classes_l2", celltype_l2_number, True)
        del data['celltype_l2']
    if 'celltype_l3' in data:
        dict_l3 = dict_multi[2]
        celltype_l3_number = [dict_l3[item] for item in data['celltype_l3']]
        data.insert(1, "classes_l3", celltype_l3_number, True)
        del data['celltype_l3']
    if 'celltype_l4' in data:
        dict_l4 = dict_multi[3]
        celltype_l4_number = [dict_l4[item] for item in data['celltype_l4']]
        data.insert(1, "classes_l4", celltype_l4_number, True)
        del data['celltype_l4']
    if 'celltype_l5' in data:
        dict_l5 = dict_multi[4]
        celltype_l5_number = [dict_l5[item] for item in data['celltype_l5']]
        data.insert(1, "classes_l5", celltype_l5_number, True)
        del data['celltype_l5']
    
    # Split data for training
    ## Split using 'celltype_l5' if it is given
    if celltype_l5 != None:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l5'])
        del data
    ## Split using 'celltype_l4' if 'celltype_l5' is not given
    elif (celltype_l5 == None) and (celltype_l4 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l4'])
        del data
    ## Split using 'celltype_l3' if 'celltype_l4' is not given
    elif (celltype_l4 == None) and (celltype_l3 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l3'])
        del data
    ## Split using 'celltype_l2' if 'celltype_l3' is not given
    elif (celltype_l3 == None) and (celltype_l2 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l2'])
        del data
    ## Split using 'celltype_l1' if 'celltype_l2' is not given
    elif (celltype_l2 == None) and (celltype_l1 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l1'])
        del data

    print(f'Train dataset contains: {len(train)} cells, it is {round(100*(len(train)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print(f'Test dataset contains: {len(test)} cells, it is {round(100*(len(test)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print()
    
    # Set target list
    target = ['classes_l1']
    if 'classes_l2' in train:
        target.append('classes_l2')
    if 'classes_l3' in train:
        target.append('classes_l3')
    if 'classes_l4' in train:
        target.append('classes_l4')
    if 'classes_l5' in train:
        target.append('classes_l5')
        
    # Variables to store history
    cash = {}
    
    # Set parameters for model training
    if os.path.isfile(os.path.join(path, model_name, 'params.txt').replace("\\","/")):
        # Load parameters used for pretraining model
        with open(os.path.join(path, model_name, 'params.txt').replace("\\","/")) as params:
            params = params.read()
            params = json.loads(params)
        print('Successfully loaded parameters')
    else:
        # Set default parameters if there is no `params.txt` in the model folder
        params = {"n_d": 8, 
                  "n_a": 8, 
                  "n_steps": 3, 
                  "n_shared": 2, 
                  "cat_emb_dim": 1, 
                  "n_independent": 2,
                  "gamma": 1.3,
                  "momentum": 0.02,
                  "optimizer_params": {"lr": 0.02},
                  "mask_type": "entmax",
                  "lambda_sparse": 0.001,
                  "device_name": "cuda",
                  "scheduler_params": {"step_size": 10, "gamma": 0.95}, 
                  "batch_size": 1024, 
                  "virtual_batch_size": 128,
                  "max_epochs": 100,
                  "patience":10}
        print('There is no `params.txt` in the model folder - Default parameters specified for the `train` function are used.')

    # Define accelerator
    if accelerator == 'auto':
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    print()
    print(f'Accelerator: {accelerator}')
    print("Start training")
        
    # Get the training features and labels
    train_target = train[target].values
    train_matrix = train[features].values
    del train
    
    # Get the validation features and labels
    test_target = test[target].values
    test_matrix = test[features].values
    del test

    aug = ClassificationSMOTE(seed = random_state)
    # Load pretrained model
    clf = TabNetMultiTaskClassifier(
                                    device_name = accelerator,
                                    optimizer_fn = optimizer_fn,
                                    scheduler_fn = scheduler_fn, 
                                    verbose = verbose,
                                    seed = random_state
                                   )
    
    clf.load_model(os.path.join(path, model_name, 'model.zip').replace("\\","/"))

    # Change model parameters
    if n_d != None:
        clf.n_d = n_d
        params['n_d'] = n_d
    else:
        clf.n_d = params['n_d']

    if n_a != None:
        clf.n_a = n_a
        params['n_a'] = n_a
    else:
        clf.n_a = params['n_a']

    if n_steps != None:
        clf.n_steps = n_steps
        params['n_steps'] = n_steps
    else:
        clf.n_steps = params['n_steps']

    if n_shared != None:
        clf.n_shared = n_shared
        params['n_shared'] = n_shared
    else:
        clf.n_shared = params['n_shared']

    if cat_emb_dim != None:
        clf.cat_emb_dim = cat_emb_dim
        params['cat_emb_dim'] = cat_emb_dim
    else:
        clf.cat_emb_dim = params['cat_emb_dim']

    if n_independent != None:
        clf.n_independent = n_independent
        params['n_independent'] = n_independent
    else:
        clf.n_independent = params['n_independent']
        
    if gamma != None:
        clf.gamma = gamma
        params['gamma'] = gamma
    else:
        clf.gamma = params['gamma']
        
    if momentum != None:
        clf.momentum = momentum
        params['momentum'] = momentum
    else:
        clf.momentum = params['momentum']
        
    if lr != None:
        clf.optimizer_params['lr'] = lr
        params['optimizer_params']['lr'] = lr
    else:
        clf.optimizer_params['lr'] = params['optimizer_params']['lr']
        
    if mask_type != None:
        clf.mask_type = mask_type
        params['mask_type'] = mask_type
    else:
        clf.mask_type = params['mask_type']
        
    if lambda_sparse != None:
        clf.lambda_sparse = lambda_sparse
        params['lambda_sparse'] = lambda_sparse
    else:
        clf.lambda_sparse = params['lambda_sparse']
        
    if step_size != None:
        clf.scheduler_params['step_size'] = step_size
        params['scheduler_params']['step_size'] = step_size
    else:
        clf.scheduler_params['step_size'] = params['scheduler_params']['step_size']
        
    if gamma_scheduler != None:
        clf.scheduler_params['gamma'] = gamma_scheduler
        params['scheduler_params']['gamma'] = gamma_scheduler
    else:
        clf.scheduler_params['gamma'] = params['scheduler_params']['gamma']
        
    if batch_size == None:
        batch_size = params['batch_size']
    
    if virtual_batch_size == None:
        virtual_batch_size = params['virtual_batch_size']
    
    if max_epochs == None:
        max_epochs = params['max_epochs']
    
    if patience == None:
        patience = params['patience']
        
    # Train model
    clf.fit(
        X_train = train_matrix,
        y_train = train_target,
        eval_set = [(train_matrix, train_target), (test_matrix, test_target)],
        eval_name = ["train", "valid"],
        eval_metric = eval_metric,
        loss_fn = loss_fn,
        max_epochs = max_epochs,
        patience = patience,
        batch_size = batch_size,
        virtual_batch_size = virtual_batch_size,
        num_workers = 0,
        drop_last = drop_last,
        warm_start = True, # For fine tuning pretrained model
        augmentations = aug
    )
        
    # Save history and parameters
    # History
    cash = clf.history.history
    cash = pd.DataFrame(cash)
    cash['epoch'] = cash.index
    cash = cash.set_index('epoch')
    cash.to_csv(os.path.join(path, model_name, 'history.csv').replace("\\","/"))
    
    # Parameters
    with open(os.path.join(path, model_name, 'params.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(params))
        
    print()
    print('Successfully saved training history and parameters')
    
    # Save tabnet model
    clf.save_model(os.path.join(path, model_name, 'model').replace("\\","/"))

    if return_model == True:
        return clf
    

# Function for hyperparameters tuning
def hyperparameter_tuning(
    adata, 
    path = '',
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    model_name = 'model_annotation_tuning',
    storage = 'model_annotation_tuning.db',
    study_name = 'study',
    load_if_exists = True,
    accelerator = 'auto',
    tune_params = 'auto',
    random_state = 0,
    num_trials = 100,
    verbose = 0, # Set to 1 to see every epoch, 0 to get None
    n_d = None, 
    n_a = None, 
    n_steps = None, 
    n_shared = None, 
    cat_emb_dim = None, 
    n_independent = None,
    gamma = None,
    momentum = None,
    lr = None, 
    lambda_sparse = None, 
    patience = None, 
    max_epochs = None,
    batch_size = None,
    virtual_batch_size = None,
    mask_type = None, 
    optimizer_fn = torch.optim.AdamW,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.CrossEntropyLoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    eval_metric = ['accuracy'],
    direction = 'maximize',
    drop_last = True
):
    
    """
    Hyperparameter tuning using the automatic model optimization framework Optuna.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to create a folder with best hyperparameters, dictionary of cell annotations and genes used for hyperparameters optimization.
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
    model_name : str, (default: 'model_annotation_tuning')
        Name of a folder to save hyperparameters, dictionary of cell annotations and genes used for hyperparameters optimization.
    storage : str, (default: 'model_annotation_tuning.db')
        Database URL. If this argument is set to None, in-memory (RAM) storage is used, and the study will not be persistent. 
        We don't recommend to use in-memory (RAM) storage to save optimization progress.
    study_name : str, (default: 'study')
        Study’s name. If this argument is set to None, a unique name is generated automatically.
    load_if_exists : bool, (default: True)
        Flag to control the behavior to handle a conflict of study names. 
        In the case where a study named study_name already exists in the storage, 
        a DuplicatedStudyError is raised if load_if_exists is set to False. 
        Otherwise, the creation of the study is skipped, and the existing one is returned.
        If the value is True, allows hyperparameter tuning to continue if interrupted (keyboard interrupt, or Windows update).
    accelerator : str, (default: 'auto')
        Type of accelerator to use in training model ('cpu', 'cuda'). Set 'auto' for automatic selection.
    tune_params : str, (default: 'auto')
        Dictionary of tunable hyperparameters with lowest and highest value and step for integer parameters.
        Example: tune_params = {"n_d": [8, 64, 4]} # first - lowest value, second - highest value, third - step.
    random_state : int, (default: 0)
        Controls the data shuffling, splitting to folds and model training.
        Pass an int for reproducible output across multiple function calls.
    num_trials : int, (default: 100)
        Number of trials to tune hyperparameters.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    n_d : int, (default: None)
        Width of the decision prediction layer. 
        Bigger values gives more capacity to the model with the risk of overfitting. 
        Values typically range from 8 to 128.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    n_a : int, (default: None) 
        Width of the attention embedding for each mask.
        Values typically range from 8 to 128.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    n_steps : int, (default: None) 
        Number of steps in the architecture.
        Values typically range from 3 to 10.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    n_shared : int, (default: None) 
        Number of shared Gated Linear Units at each step.
        Values typically range from 1 to 10.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    cat_emb_dim : int, (default: None)
        List of embeddings size for each categorical features.
        Values typically range from 1 to 10.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    n_independent : int, (default: None)
        Number of independent Gated Linear Units layers at each step. 
        Values typically range from 1 to 10.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    gamma : float, (default: None)
        This is the coefficient for feature reusage in the masks. 
        A value close to 1 will make mask selection least correlated between layers. 
        Values typically range from 1.0 to 2.0.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    momentum : float, (default: None)
        Momentum for batch normalization.
        Values typically range from 0.01 to 0.4.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    lr : float, (default: None) 
        Determines the step size at each iteration while moving toward a minimum of a loss function. 
        A large initial learning rate of 0.02  with decay is a good option.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    lambda_sparse : float, (default: None) 
        This is the extra sparsity loss coefficient. 
        The bigger this coefficient is, the sparser your model will be in terms of feature selection. 
        Depending on the difficulty of your problem, reducing this value could help.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    patience : int, (default: None)
        Number of consecutive epochs without improvement before performing early stopping.
        If patience is set to 0, then no early stopping will be performed. Values typically range from 5 to 20.
        Note that if patience is enabled, then best weights from best epoch will automatically be loaded at the end of the training.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    max_epochs : int, (default: None)
        Maximum number of epochs for training. Values typically range from 5 to 100.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    batch_size : int, (default: None)
        Number of examples per batch. Values typically range from 2 to 10 of virtual_batch_size.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    virtual_batch_size : int, (default: None)
        Size of the mini batches used for "Ghost Batch Normalization".
        'virtual_batch_size' should divide 'batch_size'. Values typically: 128, 256, 512, 1024
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    mask_type : str, (default: None)
        Either "sparsemax" or "entmax". This is the masking function to use for selecting features.
        If given, then used for the trail 0. If not specified in the list of tunable hyperparameters, then this value is used for all trails.
    optimizer_fn : func, (default: torch.optim.AdamW)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.CrossEntropyLoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    eval_metric : list, (default: ['accuracy'])
        List of evaluation metrics ('accuracy', 'balanced_accuracy', 'logloss').
        The last metric is used as the target and for early stopping.
    direction : str, (default: 'maximize')
        Directioon of optuna algorithm. 'maximize' for 'accuracy' and 'balanced_accuracy', 'minimize' for 'logloss'.
        Only for last evaluation metric given in eval_metric list.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    
    """

    
    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))

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

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     
     
    # Save gene names for future prediction
    cols = data.columns
    if celltype_l5 != None:
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4', 'celltype_l5']
    elif (celltype_l5 == None) and (celltype_l4 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4']
    elif (celltype_l4 == None) and (celltype_l3 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3']
    elif (celltype_l3 == None) and (celltype_l2 != None):
        unused = ['celltype_l1', 'celltype_l2']
    elif (celltype_l2 == None) and (celltype_l1 != None):
        unused = ['celltype_l1']
    features = [col for col in cols if col not in unused]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()

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
    
    # write a dictionary to model folder
    with open(os.path.join(path, model_name, 'dict.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(dict_multi))
    del dict_multi
    print('Successfully saved dictionary of dataset annotations')
    print()

    # Split data for training
    ## Split using 'celltype_l5' if it is given
    if celltype_l5 != None:
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['classes_l5'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['classes_l5'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['classes_l5'])
        del X_train_fold_b
    ## Split using 'celltype_l4' if 'celltype_l5' is not given
    elif (celltype_l5 == None) and (celltype_l4 != None):
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['classes_l4'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['classes_l4'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['classes_l4'])
        del X_train_fold_b
    ## Split using 'celltype_l3' if 'celltype_l4' is not given
    elif (celltype_l4 == None) and (celltype_l3 != None):
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['classes_l3'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['classes_l3'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['classes_l3'])
        del X_train_fold_b
    ## Split using 'celltype_l2' if 'celltype_l3' is not given
    elif (celltype_l3 == None) and (celltype_l2 != None):
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['classes_l2'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['classes_l2'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['classes_l2'])
        del X_train_fold_b
    ## Split using 'celltype_l1' if 'celltype_l2' is not given
    elif (celltype_l2 == None) and (celltype_l1 != None):
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['classes_l1'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['classes_l1'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['classes_l1'])
        del X_train_fold_b

    # Define fold number
    X_train_fold_1['kfold'] = 1
    X_train_fold_2['kfold'] = 2
    X_train_fold_3['kfold'] = 3
    X_train_fold_4['kfold'] = 4
    # Concatenate folds in single training dataset
    X_train = pd.concat([X_train_fold_1, X_train_fold_2, X_train_fold_3, X_train_fold_4])
    del X_train_fold_1, X_train_fold_2, X_train_fold_3, X_train_fold_4
    
    # Set target list
    target = ['classes_l1']
    if 'classes_l2' in X_train:
        target.append('classes_l2')
    if 'classes_l3' in X_train:
        target.append('classes_l3')
    if 'classes_l4' in X_train:
        target.append('classes_l4')
    if 'classes_l5' in X_train:
        target.append('classes_l5')

    # Set default parameters
    params_default = {}
    # Set default n_d
    if n_d == None:
        params_default["n_d"] = 8
    else:
        params_default["n_d"] = n_d

    # Set default n_a
    if n_a == None:
        params_default["n_a"] = 8
    else:
        params_default["n_a"] = n_a
        
    # Set default n_steps
    if n_steps == None:
        params_default["n_steps"] = 3
    else:
        params_default["n_steps"] = n_steps

    # Set default n_shared
    if n_shared == None:
        params_default["n_shared"] = 2
    else:
        params_default["n_shared"] = n_shared

    # Set default cat_emb_dim
    if cat_emb_dim == None:
        params_default["cat_emb_dim"] = 1
    else:
        params_default["cat_emb_dim"] = cat_emb_dim

    # Set default n_independent
    if n_independent == None:
        params_default["n_independent"] = 1
    else:
        params_default["n_independent"] = n_independent

    # Set default patience
    if patience == None:
        params_default["patience"] = 10
    else:
        params_default["patience"] = patience

    # Set default max_epochs
    if max_epochs == None:
        params_default["max_epochs"] = 100
    else:
        params_default["max_epochs"] = max_epochs

    # Set default batch_size
    if batch_size == None:
        params_default["batch_size"] = 1024
    else:
        params_default["batch_size"] = batch_size

    # Set default virtual_batch_size
    if virtual_batch_size == None:
        params_default["virtual_batch_size"] = 128
    else:
        params_default["virtual_batch_size"] = virtual_batch_size

    # Set default mask_type
    if mask_type == None:
        params_default["mask_type"] = 'entmax'
    else:
        params_default["mask_type"] = mask_type

    # Set default momentum
    if momentum == None:
        params_default["momentum"] = 0.02
    else:
        params_default["momentum"] = momentum

    # Set default gamma
    if gamma == None:
        params_default["gamma"] = 1.3
    else:
        params_default["gamma"] = gamma

    # Set default lambda_sparse
    if lambda_sparse == None:
        params_default["lambda_sparse"] = 0.001
    else:
        params_default["lambda_sparse"] = lambda_sparse

    # Set default lr
    if lr == None:
        params_default["lr"] = 0.01
    else:
        params_default["lr"] = lr
        
    # Set accelerator
    if accelerator == 'auto':
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    params_default["device_name"] = accelerator
    print(f'Accelerator: {accelerator}')
    print()
    
    # Set default best_score
    if os.path.isfile(os.path.join(path, model_name, 'best_score.txt').replace("\\","/")):
        with open(os.path.join(path, model_name, 'best_score.txt').replace("\\","/")) as best_score:
            best_score = best_score.read()
            best_score = json.loads(best_score)
    else:
        best_score = 0
        
    def train_params(
        train_df,
        params,
        features,
        target,
        trial,
        drop_last = drop_last,
        verbose = verbose,
        optimizer_fn = optimizer_fn,
        scheduler_fn = scheduler_fn,
        step_size = step_size,
        gamma_scheduler = gamma_scheduler,
        loss_fn = loss_fn
    ):
        # Variable to store total best score
        total_best_score = 0.0
        
        training_params = params.copy()
        del training_params["max_epochs"], training_params["patience"], training_params["batch_size"], training_params["virtual_batch_size"]
        
        for fold in range(1, 5):
            print(f"Fold {fold}:")
                
            # Get the training and validation sets
            train = X_train[X_train["kfold"] != fold]
            val = X_train[X_train["kfold"] == fold]
                
            # Get the training features and labels
            y_train = train[target].values
            train = train[features].values
                
            # Get the validation features and labels
            y_val = val[target].values
            val = val[features].values

            aug = ClassificationSMOTE(seed = random_state)
            # Create model
            clf = TabNetMultiTaskClassifier(**training_params, 
                                            optimizer_fn = optimizer_fn,
                                            scheduler_fn = scheduler_fn, 
                                            scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
                                            verbose = verbose,
                                            seed = random_state
                                           )
                
            # Train model
            clf.fit(
                X_train = train,
                y_train = y_train,
                eval_set = [(train, y_train), (val, y_val)],
                eval_name = ["train", "valid"],
                eval_metric = eval_metric,
                loss_fn = loss_fn,
                max_epochs = params.get("max_epochs"),
                patience = params.get("patience"),
                batch_size = params.get("batch_size"),
                virtual_batch_size = params.get("virtual_batch_size"),
                num_workers = 0,
                drop_last = drop_last,
                augmentations = aug
            )
            
            # Calculate summary of folds accuracies
            total_best_score += clf.best_cost

            # Get best score for a fold into trial report (used by pruner algorithm)
            trial.report(clf.best_cost, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            print()
            
        # Calculate average accuracy between folds
        total_best_score = total_best_score/4
        
        return total_best_score
    
    # Function for define objective and params
    def objective(
        trial,
        train_df,
        features,
        target,
        tune_params = tune_params,
        accelerator = accelerator,
        n_d = n_d, 
        n_a = n_a, 
        n_steps = n_steps, 
        n_shared = n_shared, 
        cat_emb_dim = cat_emb_dim, 
        n_independent = n_independent,
        gamma = gamma,
        momentum = momentum,
        lr = lr, 
        mask_type = mask_type, 
        step_size = step_size, 
        gamma_scheduler = gamma_scheduler,
        lambda_sparse = lambda_sparse, 
        patience = patience, 
        max_epochs = max_epochs,
        batch_size = batch_size,
        virtual_batch_size = virtual_batch_size,
        best_score = best_score,
        loss_fn = loss_fn
    ):
        
        # Define parameters for hyperparameter tuning        
        if tune_params == 'auto':
            
            # Set auto params if some params are given
            params = {}
            if n_d != None:
                params['n_d'] = n_d
            else:
                params['n_d'] = trial.suggest_int("n_d", 8, 128, step = 4)
                
            if n_a != None:
                params['n_a'] = n_a
            else:
                params['n_a'] = trial.suggest_int("n_a", 8, 128, step = 4)
                
            if n_steps != None:
                params['n_steps'] = n_steps
            else:
                params['n_steps'] = trial.suggest_int("n_steps", 1, 10, step = 1)
                
            if n_shared != None:
                params['n_shared'] = n_shared
            else:
                params['n_shared'] = trial.suggest_int("n_shared", 1, 10, step = 1)
                
            if cat_emb_dim != None:
                params['cat_emb_dim'] = cat_emb_dim
            else:
                params['cat_emb_dim'] = trial.suggest_int("cat_emb_dim", 1, 10, step = 1)
                
            if n_independent != None:
                params['n_independent'] = n_independent
            else:
                params['n_independent'] = trial.suggest_int("n_independent", 1, 10, step = 1)
                
            if gamma != None:
                params['gamma'] = gamma
            else:
                params['gamma'] = trial.suggest_float("gamma", 1, 2)
                
            if momentum != None:
                params['momentum'] = momentum
            else:
                params['momentum'] = trial.suggest_float("momentum", 0.01, 0.4)
                
            if lr != None:
                params['optimizer_params'] = {"lr": lr}
            else:
                params['optimizer_params'] = {"lr": trial.suggest_float("lr", 0.0001, 0.5)}
                
            if mask_type != None:
                params['mask_type'] = mask_type
            else:
                params['mask_type'] = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
                
            if lambda_sparse != None:
                params['lambda_sparse'] = lambda_sparse
            else:
                params['lambda_sparse'] = trial.suggest_float("lambda_sparse", 1e-4, 5e-2, log=True)
                
            if patience != None:
                params['patience'] = patience
            else:
                params['patience'] = trial.suggest_int("patience", 5, 20, step = 5)
                
            if max_epochs != None:
                params['max_epochs'] = max_epochs
            else:
                params['max_epochs'] = trial.suggest_int("max_epochs", 5, 100, step = 5)
            
            if virtual_batch_size != None:
                params['virtual_batch_size'] = virtual_batch_size
            else:
                params['virtual_batch_size'] = trial.suggest_categorical("virtual_batch_size", [128, 256, 512, 1024])
              
            if batch_size != None:
                params['batch_size'] = batch_size
            elif params['virtual_batch_size'] == 1024:
                params['batch_size'] = trial.suggest_int("batch_size", 2*params['virtual_batch_size'], 4*params['virtual_batch_size'], step = params['virtual_batch_size'])
            elif params['virtual_batch_size'] == 512:
                params['batch_size'] = trial.suggest_int("batch_size", 2*params['virtual_batch_size'], 6*params['virtual_batch_size'], step = params['virtual_batch_size'])
            else:
                params['batch_size'] = trial.suggest_int("batch_size", 2*params['virtual_batch_size'], 10*params['virtual_batch_size'], step = params['virtual_batch_size'])
                
        else: 
            params = {}
            ## Set "n_d"
            if "n_d" in tune_params:
                params["n_d"] = trial.suggest_int("n_d", tune_params["n_d"][0], tune_params["n_d"][1], step = tune_params["n_d"][2])
            elif n_d != None:
                params["n_d"] = n_d
            else:
                params["n_d"] = 8
                
            ## Set "n_a"
            if "n_a" in tune_params:
                params["n_a"] = trial.suggest_int("n_a", tune_params["n_a"][0], tune_params["n_a"][1], step = tune_params["n_a"][2])
            elif n_a != None:
                params["n_a"] = n_a
            else:
                params["n_a"] = 8
                
            ## Set "n_steps"
            if "n_steps" in tune_params:
                params["n_steps"] = trial.suggest_int("n_steps", tune_params["n_steps"][0], tune_params["n_steps"][1], step = tune_params["n_steps"][2])
            elif n_steps != None:
                params["n_steps"] = n_steps
            else:
                params["n_steps"] = 3
                
            ## Set 'n_shared'
            if "n_shared" in tune_params:
                params["n_shared"] = trial.suggest_int("n_shared", tune_params["n_shared"][0], tune_params["n_shared"][1], step = tune_params["n_shared"][2])
            elif n_shared != None:
                params["n_shared"] = n_shared
            else:
                params["n_shared"] = 2
                
            ## Set 'cat_emb_dim'
            if "cat_emb_dim" in tune_params:
                params["cat_emb_dim"] = trial.suggest_int("cat_emb_dim", tune_params["cat_emb_dim"][0], tune_params["cat_emb_dim"][1], step = tune_params["cat_emb_dim"][2])
            elif cat_emb_dim != None:
                params["cat_emb_dim"] = cat_emb_dim
            else:
                params["cat_emb_dim"] = 1
                
            ## Set 'n_independent'
            if "n_independent" in tune_params:
                params["n_independent"] = trial.suggest_int("n_independent", tune_params["n_independent"][0], tune_params["n_independent"][1], step = tune_params["n_independent"][2])
            elif n_independent != None:
                params["n_independent"] = n_independent
            else:
                params["n_independent"] = 2
                
            ## Set 'patience'
            if "patience" in tune_params:
                params["patience"] = trial.suggest_int("patience", tune_params["patience"][0], tune_params["patience"][1], step = tune_params["patience"][2])
            elif patience != None:
                params["patience"] = patience
            else:
                params["patience"] = 10
                
            ## Set 'max_epochs'
            if "max_epochs" in tune_params:
                params["max_epochs"] = trial.suggest_int("max_epochs", tune_params["max_epochs"][0], tune_params["max_epochs"][1], step = tune_params["max_epochs"][2])
            elif max_epochs != None:
                params["max_epochs"] = max_epochs
            else:
                params["max_epochs"] = 100

            ## Set 'virtual_batch_size'
            if "virtual_batch_size" in tune_params:
                params["virtual_batch_size"] = trial.suggest_int("virtual_batch_size", tune_params["virtual_batch_size"][0], tune_params["virtual_batch_size"][1], step = tune_params["virtual_batch_size"][2])
            elif virtual_batch_size != None:
                params["virtual_batch_size"] = virtual_batch_size
            else:
                params["virtual_batch_size"] = 128
            
            ## Set 'batch_size'
            if "batch_size" in tune_params:
                i = 2
                j = 10
                while i*params['virtual_batch_size'] < params['batch_size'][0]:
                    i += 1
                while j*params['virtual_batch_size'] > params['batch_size'][1]:
                    j -= 1
                params["batch_size"] = trial.suggest_int("batch_size", i*params['virtual_batch_size'], j*params['virtual_batch_size'], step = params['virtual_batch_size'])
            elif batch_size != None:
                params["batch_size"] = batch_size
            else:
                params["batch_size"] = 1024
                
            ## Set 'mask_type'
            if "mask_type" in tune_params:
                params["mask_type"] = trial.suggest_categorical("mask_type", [tune_params["mask_type"][0], tune_params["mask_type"][1]])
            elif mask_type != None:
                params["mask_type"] = mask_type
            else:
                params["mask_type"] = 'entmax'
                
            ## Set 'momentum'
            if "momentum" in tune_params:
                params["momentum"] = trial.suggest_float("momentum", tune_params["momentum"][0], tune_params["momentum"][1]) # float without step
            elif momentum != None:
                params["momentum"] = momentum
            else:
                params["momentum"] = 0.02
                                
            ## Set 'lr'
            if "lr" in tune_params:
                params["optimizer_params"] = {"lr": trial.suggest_float("lr", tune_params["lr"][0], tune_params["lr"][1])}
            elif lr != None:
                params["optimizer_params"] = {"lr": lr}
            else:
                params["optimizer_params"] = {"lr": 0.01}
                
            ## Set 'gamma'
            if "gamma" in tune_params:
                params["gamma"] = trial.suggest_float("gamma", tune_params["gamma"][0], tune_params["gamma"][1])
            elif gamma != None:
                params["gamma"] = gamma
            else:
                params["gamma"] = 1.3
                
            ## Set 'lambda_sparse'
            if "lambda_sparse" in tune_params:
                params["lambda_sparse"] = trial.suggest_float("lambda_sparse", tune_params["lambda_sparse"][0], tune_params["lambda_sparse"][1], log=True)
            elif lambda_sparse != None:
                params["lambda_sparse"] = lambda_sparse
            else:
                params["lambda_sparse"] = 0.001

        # Set accelerator
        if accelerator == 'auto':
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        params["device_name"] = accelerator
        
        score = train_params(
            train_df = train_df,
            params = params,
            features = features,
            target = target,
            verbose = verbose,
            trial = trial,
            loss_fn = loss_fn
        )
        
        if score > best_score:
            best_score = score.copy()
            del params['device_name']
            # write best_params to model folder
            with open(os.path.join(path, model_name, 'best_params.txt').replace("\\","/"), 'w') as f: 
                f.write(json.dumps(params))
            with open(os.path.join(path, model_name, 'best_score.txt').replace("\\","/"), 'w') as f: 
                f.write(json.dumps(best_score))
                
        return score
        

    # Function for hyperparameter search
    def hyperparameter_search(
        train_df,
        features,
        target,
        num_trials,
        storage = storage,
        model_name = model_name,
        load_if_exists = load_if_exists
    ):  
        # Get the objective
        objective_ = functools.partial(
            objective,
            train_df = train_df,
            features = features,
            target = target
        )
        
        # Create study using optuna
        sampler = optuna.samplers.TPESampler(seed = random_state)
        if storage != None:
            storage = os.path.join(path, model_name, storage).replace("\\","/")
            storage = optuna.storages.RDBStorage(url = f"sqlite:///{storage}",
                                                 heartbeat_interval = 60,
                                                 failed_trial_callback = optuna.storages.RetryFailedTrialCallback(max_retry = 1),
                                                )
            
        study = optuna.create_study(direction = direction, 
                                    storage = storage,
                                    study_name = model_name,
                                    load_if_exists = load_if_exists,
                                    sampler = sampler,
                                    pruner = optuna.pruners.HyperbandPruner(min_resource = 1, 
                                                                            max_resource = 'auto', 
                                                                            reduction_factor = 3
                                                                           )
                                   )
        
        # Enqueue a trial which uses the default parameters
        if not study.trials:
            study.enqueue_trial(params_default)
            num_trials = num_trials
        # Decrease number of trials if optuna interrupts
        elif study.trials[-1].state == optuna.trial.TrialState.FAIL: 
            num_trials = num_trials - len(study.trials)

        # Optimize study using optuna
        study.optimize(objective_, 
                       n_trials = num_trials)
    
        return study.best_params

    best_params = hyperparameter_search(train_df = X_train,
                                        features = features,
                                        target = target,
                                        num_trials = num_trials
                                       )
    
    # Save best hyperparameters
    best_params["optimizer_params"] = {"lr": best_params['lr']}
    del best_params['lr']
    with open(os.path.join(path, model_name, 'best_params.txt').replace("\\","/"), 'w') as f: 
                f.write(json.dumps(best_params))
    print("Successfully saved best hyperparameters")
    print()
        
    # write best_params to model folder
    #with open(os.path.join(path, model_name, 'best_params.txt').replace("\\","/"), 'w') as f: 
    #    f.write(json.dumps(best_params))
    
    print(f"Best hyperparameters: {best_params}")
    return best_params


# Function for training model using parameters tuned by scparadise.scadam.tune
def train_tuned(
    adata, 
    path = '',
    path_tuned = '', 
    celltype_l1 = None, 
    celltype_l2 = None, 
    celltype_l3 = None, 
    celltype_l4 = None, 
    celltype_l5 = None,
    model_name = 'model_annotation_tuned',
    accelerator = 'auto',
    random_state = 0,
    test_size = 0.1,
    optimizer_fn = torch.optim.AdamW,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.CrossEntropyLoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    verbose = True, 
    eval_metric = ['accuracy'],
    drop_last = True,
    return_model = False,
    from_unsupervised = True,
    pretraining_ratio = 0.5    
):
    '''
    Train the scAdam model using parameters tuned by the 'scparadise.scadam.hyperparameter_tuning' function.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path : str, path object
        Path to create a folder with model, training history, dictionary of cell annotations and genes used for training.
    path_tuned : str, path object
        Path to folder with tuned parameters by scparadise.scadam.hyperparameter_tuning function.
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
    model_name : str, (default: 'model_annotation_tuned')
        Name of a folder to save model.
    accelerator : str, (default: 'auto')
        Type of accelerator to use in training model ('cpu', 'cuda'). Set 'auto' for automatic selection.
    random_state : int, (default: 0)
        Controls the data shuffling, splitting to folds and model training.
        Pass an int for reproducible output across multiple function calls.
    test_size : float or int, (default: 0.1)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test cells.
    optimizer_fn : func, (default: torch.optim.AdamW)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.CrossEntropyLoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    eval_metric : list, (default: 'accuracy')
        List of evaluation metrics ('accuracy', 'balanced_accuracy', 'logloss').
        The last metric is used for early stopping.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    return_model : bool, (default: False)
        Return model after training or not.
    from_unsupervised : bool, (default: True)
        Use a previously self supervised model as starting weights. 
        Supervised model training included in function.
    pretraining_ratio : float, (default: 0.5)
        Between 0 and 1, percentage of feature to mask for reconstruction.
        Used for supervised model training.    
    '''
    
    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))

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

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     
     
    # Save gene names for future prediction
    cols = data.columns
    if celltype_l5 != None:
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4', 'celltype_l5']
    elif (celltype_l5 == None) and (celltype_l4 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3', 'celltype_l4']
    elif (celltype_l4 == None) and (celltype_l3 != None):
        unused = ['celltype_l1', 'celltype_l2', 'celltype_l3']
    elif (celltype_l3 == None) and (celltype_l2 != None):
        unused = ['celltype_l1', 'celltype_l2']
    elif (celltype_l2 == None) and (celltype_l1 != None):
        unused = ['celltype_l1']
    features = [col for col in cols if col not in unused]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()

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
    
    # write a dictionary to model folder
    with open(os.path.join(path, model_name, 'dict.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(dict_multi))
    del dict_multi
    print('Successfully saved dictionary of dataset annotations')
    print()
    
    # Split data for training
    ## Split using 'celltype_l5' if it is given
    if celltype_l5 != None:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l5'])
        del data
    ## Split using 'celltype_l4' if 'celltype_l5' is not given
    elif (celltype_l5 == None) and (celltype_l4 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l4'])
        del data
    ## Split using 'celltype_l3' if 'celltype_l4' is not given
    elif (celltype_l4 == None) and (celltype_l3 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l3'])
        del data
    ## Split using 'celltype_l2' if 'celltype_l3' is not given
    elif (celltype_l3 == None) and (celltype_l2 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l2'])
        del data
    ## Split using 'celltype_l1' if 'celltype_l2' is not given
    elif (celltype_l2 == None) and (celltype_l1 != None):
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['classes_l1'])
        del data

    
    print(f'Train dataset contains: {len(train)} cells, it is {round(100*(len(train)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print(f'Test dataset contains: {len(test)} cells, it is {round(100*(len(test)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print()
    
    # Set target list
    target = ['classes_l1']
    if 'classes_l2' in train:
        target.append('classes_l2')
    if 'classes_l3' in train:
        target.append('classes_l3')
    if 'classes_l4' in train:
        target.append('classes_l4')
    if 'classes_l5' in train:
        target.append('classes_l5')
        
    # Variables to store history, explainability and total best score
    cash = {}
    explains = {}
    
    # Create parameters for learning model
    with open(os.path.join(path_tuned, 'best_params.txt')) as params:
        params = params.read()
        params = json.loads(params)
    
    # Define accelerator
    if accelerator == 'auto':
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    params["device_name"] = accelerator
    print(f'Accelerator: {accelerator}')
            
    # Get the training features and labels
    train_target = train[target].values
    train_matrix = train[features].values
    del train
    
    # Get the validation features and labels
    test_target = test[target].values
    test_matrix = test[features].values
    del test

    # Print params tuned by scparadise.scadam.tune
    print(f'Start training with following hyperparameters: {params}')
    print()
    
    # Modify params previously saved by scparadise.scadam.tune function
    max_epochs = params["max_epochs"]
    patience = params["patience"]
    batch_size = params["batch_size"]
    virtual_batch_size = params["virtual_batch_size"]
    del params["max_epochs"], params["patience"], params["batch_size"], params["virtual_batch_size"]

    # Augmentation
    aug = ClassificationSMOTE(seed = random_state)
    
    # Create unsupervised model
    if from_unsupervised:
        unsupervised_model = TabNetPretrainer(
            **params,
            optimizer_fn = optimizer_fn,
            scheduler_fn = scheduler_fn, 
            scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
            verbose = verbose,
            seed = random_state
        )
        
        unsupervised_model.fit(
            X_train=train_matrix,
            eval_set=[test_matrix],
            pretraining_ratio=pretraining_ratio,
            #loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last
        )
        
    # Create model
    clf = TabNetMultiTaskClassifier(
        **params,
        optimizer_fn = optimizer_fn,
        scheduler_fn = scheduler_fn, 
        scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
        verbose = verbose,
        seed = random_state
    )
        
    # Train model
    if from_unsupervised:
        clf.fit(
            X_train = train_matrix,
            y_train = train_target,
            eval_set = [(train_matrix, train_target), (test_matrix, test_target)],
            eval_name = ["train", "valid"],
            eval_metric = eval_metric,
            loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last,
            augmentations = aug,
            from_unsupervised=unsupervised_model
        )
    else:
        clf.fit(
            X_train = train_matrix,
            y_train = train_target,
            eval_set = [(train_matrix, train_target), (test_matrix, test_target)],
            eval_name = ["train", "valid"],
            eval_metric = eval_metric,
            loss_fn = loss_fn,
            max_epochs = max_epochs,
            patience = patience,
            batch_size = batch_size,
            virtual_batch_size = virtual_batch_size,
            num_workers = 0,
            drop_last = drop_last,
            augmentations = aug
        )
        
    # Save history and parameters
    # History
    cash = clf.history.history
    cash = pd.DataFrame(cash)
    cash['epoch'] = cash.index
    cash = cash.set_index('epoch')
    cash.to_csv(os.path.join(path, model_name, 'history.csv').replace("\\","/"))
    
    # Parameters
    params["scheduler_params"] = {"step_size": step_size, "gamma": gamma_scheduler}
    params["batch_size"] = batch_size
    params["virtual_batch_size"] = virtual_batch_size
    with open(os.path.join(path, model_name, 'params.txt').replace("\\","/"), 'w') as f: 
        f.write(json.dumps(params))
        
    print()
    print('Successfully saved training history and parameters')
    
    # Save tabnet model
    clf.save_model(os.path.join(path, model_name, 'model').replace("\\","/"))

    if return_model == True:
        return clf


# Function for prediction cell types using trained model
def predict(
    adata, 
    path_model = ''
):
    '''
    Predict cell types and cell type probilities using pretrained scAdam model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path_model : str, path object
        Path to the folder containing the trained scAdam model.
        
    '''
    
    # load genes of trained model
    features = pd.read_csv(os.path.join(path_model, 'genes.csv').replace("\\","/"))    
    features = list(features['feature_name'])
    print('Successfully loaded list of genes used for training model')
    print()
    
    # Create dataset for prediction
    data_genes = adata.raw.var_names.tolist()
    data_predict = pd.DataFrame(adata.raw.X.toarray(), columns = data_genes)
    sorted_val_dataset = pd.DataFrame(index = [i for i in range(0, len(adata.obs_names))])
    for column in features:
        if column in data_genes:
            sorted_val_dataset[column] = data_predict[column]
        else:
            sorted_val_dataset[column] = 0

    # Load dictionary of trained cell types
    with open(os.path.join(path_model, 'dict.txt')) as dict:
        dict = dict.read()
        dict_multi = json.loads(dict)
    print('Successfully loaded dictionary of dataset annotations')
    print()

    # Load pretrained model
    loaded_model = TabNetMultiTaskClassifier()
    for file in os.listdir(path_model):
        if file.endswith('.zip'):
            loaded_model.load_model(os.path.join(path_model, file).replace("\\","/"))
            print('Successfully loaded model')
            print()

    # Predict cell types
    predictions = loaded_model.predict(sorted_val_dataset.values)
    # Get prediction probabilities
    probabilities = loaded_model.predict_proba(sorted_val_dataset.values)

    # Define get_key function for dictionaries
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
            
    # Add predictions and probabilities to adata
    for i in range(len(dict_multi)):
        prediction_i = [get_key(dict_multi[i], prediction) for prediction in predictions[i].astype(dtype=int)]
        adata.obs['pred_celltype_l' + f'{i+1}'] = prediction_i
        probabilities_i = probabilities[i]
        probabilities__i = []
        for j in range(len(probabilities_i)):
            probabilities__i.append(max(probabilities_i[j]))
        adata.obs['prob_celltype_l' + f'{i+1}'] = probabilities__i
        print(f'Successfully added predicted celltype_l{i+1} and cell type probabilities')
    return adata


# Function to display available models in github
def available_models(

):
    '''
    Download dataframe with available pretrained scAdam models.
    '''
    models = pd.read_csv('https://raw.githubusercontent.com/Chechekhins/scParadise/main/scadam_available_models.csv', sep=',')
    return models


# Function for downloading tuned pretrained models from github
def download_model(
    model_name = '',
    save_path = ''    
):
    '''
    Download pretrained tuned model for highly accurate cell type annotation.
    
    Parameters
    ----------
    model_name : str
        Name of the model from column 'model' from scparadise.scadam.available_models().
    save_path : str, path object
        Path to save trained scAdam model.
        
    '''

    # Create new directory with model
    save = os.path.join(save_path, model_name+'_scAdam').replace("\\","/")
    if not os.path.exists(save):
        os.makedirs(save)

    # Download content of model
    fs = fsspec.filesystem("github", org="Chechekhins", repo="scParadise")
    fs.get(fs.ls(os.path.join('models_scadam', model_name+'_scAdam').replace("\\","/")), save)
    