import anndata
import os 
import pandas as pd
import numpy as np
import muon as mu
import torch 
import json 
import fsspec
import optuna
import functools
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.augmentations import RegressionSMOTE
from sklearn.model_selection import train_test_split
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")


# Function for training model
def train(
    mdata, 
    path = '',
    rna_modality_name = 'rna',
    second_modality_name = 'adt',
    detailed_annotation = None, 
    model_name = 'model_regression',
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
    patience = 20, 
    max_epochs = 200,
    batch_size = 1024,
    virtual_batch_size = 128,
    mask_type = 'entmax', 
    eval_metric = ['rmse'],
    optimizer_fn = torch.optim.Adam,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.MSELoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    verbose = True,
    drop_last = True,
    return_model = False
):

    '''
    Train custom scEve model using MuData object with different modalities.
    
    Parameters
    ----------
    mdata : MuData
        MuData object.
    rna_modality_name : str, (default: 'rna')
        Name of RNA (GEX) modality in MuData object.
    second_modality_name : str, (default: 'adt')
        Name of protein (ADT) or ATAC-seq modality in MuData object.
    path : str, path object
        Path to create a folder with model, training history, dictionary of cell annotations and genes used for training.
    detailed_annotation : str, (default: None)
        The most detailed level of cell annotation. Key in mdata.obs dataframe.
        If given may increase model evaluation score.
    model_name : str, (default: 'model_regression')
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
    max_epochs : int, (default: 200)
        Maximum number of epochs for training.
    batch_size : int, (default: 1024)
        Number of examples per batch. 
        It is highly recomended to tune this parameter.
    virtual_batch_size : int, (default: 128)
        Size of the mini batches used for "Ghost Batch Normalization".
        'virtual_batch_size' should divide 'batch_size'.
    mask_type : str, (default: 'entmax')
        Either "sparsemax" or "entmax". This is the masking function to use for selecting features.
    optimizer_fn : func, (default: torch.optim.Adam)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.MSELoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    eval_metric : list, (default: ['rmse'])
        List of evaluation metrics ('mse', 'mae', 'rmse', 'rmsle').
        The last metric is used as the target and for early stopping.
        Mean Squared Logarithmic Error (rmsle) cannot be used when targets contain negative values.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    return_model : bool, (default: False)
        Return model after training or not.
        
    '''

    
    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))

    # Create datasets of rna and adt for model training
    data_rna = pd.DataFrame(data=mdata[rna_modality_name].X.toarray(), columns=mdata[rna_modality_name].var_names)
    data_adt = pd.DataFrame(data=mdata[second_modality_name].X.toarray(), columns=mdata[second_modality_name].var_names)

    # Define names of proteins 
    target = [marker for marker in data_adt.columns]

    # Join rna and protein datasets
    data = data_rna.join(data_adt)
    del data_rna, data_adt
    
    # Add celltype to data
    if detailed_annotation != None:
        data['detailed_annotation'] = mdata.obs[detailed_annotation].values

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     

    # Save gene names for future imputation
    cols = data.columns
    features = [col for col in cols if col not in target + ['detailed_annotation']]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()
    pd.DataFrame({'feature_name':target}).to_csv(os.path.join(path, model_name, 'proteins.csv').replace("\\","/"), index=False)
    print('Successfully saved proteins names for training model')
    print()
    
    # Split data for training
    if detailed_annotation != None:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['detailed_annotation'])
    else:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state)
    del data

    print(f'Train dataset contains: {len(train)} cells, it is {round(100*(len(train)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print(f'Test dataset contains: {len(test)} cells, it is {round(100*(len(test)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print()
    
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
    
    print("Start training")
        
    # Get the training features and labels
    train_target = train[target].values
    train_matrix = train[features].values
    del train
    
    # Get the validation features and labels
    test_target = test[target].values
    test_matrix = test[features].values
    del test

    # Augmentation
    aug = RegressionSMOTE(seed = random_state)
    
    # Create model
    clf = TabNetRegressor(**params,
                            optimizer_fn = optimizer_fn,
                            scheduler_fn = scheduler_fn, 
                            scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
                            verbose = verbose,
                            seed = random_state
                         )
        
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
        augmentations = aug
    )
    
    # Store data for explainability
    # explains = clf.explain(test_matrix)
        
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

    # Save tabnet model
    clf.save_model(os.path.join(path, model_name, 'model').replace("\\","/"))

    if return_model == True:
        return clf


# Function for hyperparameters tuning
def hyperparameter_tuning(
    mdata, 
    path = '',
    rna_modality_name = 'rna',
    second_modality_name = 'adt',
    detailed_annotation = None, 
    model_name = 'model_regression_tuning',
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
    optimizer_fn = torch.optim.Adam,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.MSELoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    eval_metric = ['rmse'],
    drop_last = True
):
    
    """
    Hyperparameter tuning using the automatic model optimization framework Optuna.

    Parameters
    ----------
    mdata : MuData
        MuData object.
    rna_modality_name : str, (default: 'rna')
        Name of RNA (GEX) modality in MuData object.
    second_modality_name : str, (default: 'adt')
        Name of protein (ADT) or ATAC-seq (ATAC) modality in MuData object.
    path : str, path object
        Path to create a folder with best hyperparameters, dictionary of cell annotations and genes used for hyperparameters optimization.
    detailed_annotation : str, (default: None)
        The most detailed level of cell annotation. Key in mdata.obs dataframe.
        If given may increase model evaluation score.
    model_name : str, (default: 'model_regression_tuning')
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
        Maximum number of epochs for training. Values typically range from 5 to 200.
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
    optimizer_fn : func, (default: torch.optim.Adam)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.MSELoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    eval_metric : list, (default: ['rmse'])
        List of evaluation metrics ('mse', 'mae', 'rmse', 'rmsle').
        The last metric is used as the target and for early stopping.
        Mean Squared Logarithmic Error (rmsle) cannot be used when targets contain negative values.
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    """

    
    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))

    # Create datasets of rna and adt for model training
    data_rna = pd.DataFrame(data=mdata[rna_modality_name].X.toarray(), columns=mdata[rna_modality_name].var_names)
    data_adt = pd.DataFrame(data=mdata[second_modality_name].X.toarray(), columns=mdata[second_modality_name].var_names)

    # Define names of proteins 
    target = [marker for marker in data_adt.columns]

    # Join rna and protein datasets
    data = data_rna.join(data_adt)
    del data_rna, data_adt
    
    # Add celltype to data
    if detailed_annotation != None:
        data['detailed_annotation'] = mdata.obs[detailed_annotation].values

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     

    # Save gene names for future imputation
    cols = data.columns
    features = [col for col in cols if col not in target + ['detailed_annotation']]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()
    pd.DataFrame({'feature_name':target}).to_csv(os.path.join(path, model_name, 'proteins.csv').replace("\\","/"), index=False)
    print('Successfully saved proteins names for training model')
    print()
    
    # Split data for training
    # Split data for training
    if detailed_annotation != None:
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = data['detailed_annotation'])
        del data
        X_train_fold_1, X_train_fold_2 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_a['detailed_annotation'])
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state, 
                                                          stratify = X_train_fold_b['detailed_annotation'])
        del X_train_fold_b
    else:
        X_train_fold_a, X_train_fold_b = train_test_split(data, 
                                                          test_size = 0.5, 
                                                          random_state = random_state)
        del data
        X_train_fold_1, X_train_fold_1 = train_test_split(X_train_fold_a, 
                                                          test_size = 0.5, 
                                                          random_state = random_state)
        del X_train_fold_a
        X_train_fold_3, X_train_fold_4 = train_test_split(X_train_fold_b, 
                                                          test_size = 0.5, 
                                                          random_state = random_state)
        del X_train_fold_b
        

    # Define fold number
    X_train_fold_1['kfold'] = 1
    X_train_fold_2['kfold'] = 2
    X_train_fold_3['kfold'] = 3
    X_train_fold_4['kfold'] = 4
    # Concatenate folds in single training dataset
    X_train = pd.concat([X_train_fold_1, X_train_fold_2, X_train_fold_3, X_train_fold_4])
    del X_train_fold_1, X_train_fold_2, X_train_fold_3, X_train_fold_4
    
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
        best_score = 1
    
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

            aug = RegressionSMOTE(seed = random_state)
            # Create model
            clf = TabNetRegressor(**training_params, 
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
                params['patience'] = trial.suggest_int("patience", 5, 30, step = 5)
                
            if max_epochs != None:
                params['max_epochs'] = max_epochs
            else:
                params['max_epochs'] = trial.suggest_int("max_epochs", 5, 300, step = 5)
                
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
                params["max_epochs"] = 25

            ## Set 'virtual_batch_size'
            if "virtual_batch_size" in tune_params:
                params["virtual_batch_size"] = trial.suggest_int("virtual_batch_size", tune_params["virtual_batch_size"][0], tune_params["virtual_batch_size"][1], step = tune_params["virtual_batch_size"][2])
            elif virtual_batch_size != None:
                params["virtual_batch_size"] = virtual_batch_size
            else:
                params["virtual_batch_size"] = 256
            
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
        
        if score < best_score:
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
            
        study = optuna.create_study(direction = 'minimize', 
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
                       n_trials = num_trials)#, n_jobs=-1)
    
        return study.best_params

    best_params = hyperparameter_search(train_df = X_train,
                                        features = features,
                                        target = target,
                                        num_trials = num_trials
                                       )
    print()

    # Save best hyperparameters
    best_params["optimizer_params"] = {"lr": best_params['lr']}
    del best_params['lr']
    with open(os.path.join(path, model_name, 'best_params.txt').replace("\\","/"), 'w') as f: 
                f.write(json.dumps(best_params))
    print("Successfully saved best hyperparameters")
    print()
    
    print(f"Best hyperparameters: {best_params}")
    return best_params


# Function for training model using parameters tuned by scparadise.scadam.tune
def train_tuned(
    mdata, 
    path = '',
    path_tuned = '',
    rna_modality_name = 'rna',
    second_modality_name = 'adt',
    detailed_annotation = None,
    model_name = 'model_regression_tuned',
    accelerator = 'auto',
    random_state = 0,
    test_size = 0.1,
    optimizer_fn = torch.optim.Adam,
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    loss_fn = torch.nn.MSELoss(),
    step_size = 10, 
    gamma_scheduler = 0.95,
    verbose = True, 
    eval_metric = ['accuracy'],
    drop_last = True,
    return_model = False
):
    '''
    Train the scEve model using parameters tuned by the 'scparadise.sceve.hyperparameter_tuning' function.
    
    Parameters
    ----------
    mdata : MuData
        MuData object.
    rna_modality_name : str, (default: 'rna')
        Name of RNA (GEX) modality in MuData object.
    second_modality_name : str, (default: 'adt')
        Name of protein (ADT) or ATAC-seq modality in MuData object.
    path : str, path object
        Path to create a folder with model, training history, dictionary of cell annotations and genes used for training.
    path_tuned : str, path object
        Path to folder with tuned parameters by scparadise.scadam.hyperparameter_tuning function.
    detailed_annotation : str, (default: None)
        The most detailed level of cell annotation. Key in mdata.obs dataframe.
        If given may increase model evaluation score.
    model_name : str, (default: 'model_regression_tuned')
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
    optimizer_fn : func, (default: torch.optim.Adam)
        Pytorch Optimizer function.
    scheduler_fn : func, (default: torch.optim.lr_scheduler.StepLR)
        Pytorch Scheduler to change learning rates during training.
    loss_fn : torch.loss function (default: torch.nn.MSELoss)
        Loss function for training.
    step_size : int, (default: 10)
        Scheduler learning rate decay.
    gamma_scheduler : float, (default: 0.95) 
        Multiplicative factor of scheduler learning rate decay. 
        step_size and gamma_scheduler are used in dictionary of parameters to apply to the scheduler_fn.
    verbose : int (0 or 1), bool (True or False), (default: True)
        Show progress bar for each epoch during training. 
        Set to 1 or 'True' to see every epoch progress, 0 or 'False' to get None.
    eval_metric : list, (default: ['rmse'])
        List of evaluation metrics ('mse', 'mae', 'rmse', 'rmsle').
        The last metric is used for early stopping.
        Mean Squared Logarithmic Error (rmsle) cannot be used when targets contain negative values.        
    drop_last : bool, (default: True)
        Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    return_model : bool, (default: False)
        Return model after training or not. 
        
    '''
    
    # Create new directory with model and list of genes
    if not os.path.exists(os.path.join(path, model_name).replace("\\","/")):
        os.makedirs(os.path.join(path, model_name).replace("\\","/"))

    # Create datasets of rna and adt for model training
    data_rna = pd.DataFrame(data=mdata[rna_modality_name].X.toarray(), columns=mdata[rna_modality_name].var_names)
    data_adt = pd.DataFrame(data=mdata[second_modality_name].X.toarray(), columns=mdata[second_modality_name].var_names)

    # Define names of proteins 
    target = [marker for marker in data_adt.columns]

    # Join rna and protein datasets
    data = data_rna.join(data_adt)
    del data_rna, data_adt
    
    # Add celltype to data
    if detailed_annotation != None:
        data['detailed_annotation'] = mdata.obs[detailed_annotation].values

    # Shuffle dataset by genes and cells
    data = data.sample(frac=1, axis=1, random_state = random_state).sample(frac=1, random_state = random_state)     

    # Save gene names for future imputation
    cols = data.columns
    features = [col for col in cols if col not in target + ['detailed_annotation']]
    pd.DataFrame({'feature_name':features}).to_csv(os.path.join(path, model_name, 'genes.csv').replace("\\","/"), index=False)
    print('Successfully saved genes names for training model')
    print()
    pd.DataFrame({'feature_name':target}).to_csv(os.path.join(path, model_name, 'proteins.csv').replace("\\","/"), index=False)
    print('Successfully saved proteins names for training model')
    print()
    
    # Split data for training
    if detailed_annotation != None:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state, 
                                       stratify = data['detailed_annotation'])
    else:
        train, test = train_test_split(data, 
                                       test_size = test_size, 
                                       random_state = random_state)
    del data

    print(f'Train dataset contains: {len(train)} cells, it is {round(100*(len(train)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print(f'Test dataset contains: {len(test)} cells, it is {round(100*(len(test)/(len(train) + len(test))), ndigits=2)} % of input dataset')
    print()
    
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
    aug = RegressionSMOTE(seed = random_state)
    
    # Create model
    clf = TabNetRegressor(**params,
                          optimizer_fn = optimizer_fn,
                          scheduler_fn = scheduler_fn, 
                          scheduler_params = {"step_size": step_size, "gamma": gamma_scheduler},
                          verbose = verbose,
                          seed = random_state
                         )
    
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
    
    # Save tabnet model
    clf.save_model(os.path.join(path, model_name, 'model').replace("\\","/"))

    if return_model == True:
        return clf


# Function for imputation proteins using trained model
def predict(
    adata, 
    path_model = '',
    rna_modality_name = 'rna',
    second_modality_name = 'adt',
    return_mdata = True
):
    '''
    Predict (impute) the second modality in cells using the pretrained scEve model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    path_model : str, path object
        Path to the folder containing the trained scEve model.
    rna_modality_name : str, (default: 'rna')
        Name of RNA (GEX) modality in MuData object.
    second_modality_name : str, (default: 'adt')
        Name of protein (ADT) or ATAC-seq modality in MuData object.      
    return_mdata : bool, (default: True)
        If set 'True' return MuData object. If set 'False' return AnnData object with predicted (imputed) modality.
        
    '''
    
    # load genes of trained model
    features = pd.read_csv(os.path.join(path_model, 'genes.csv'))
    features = list(features['feature_name'])
    print('Successfully loaded list of genes used for training model')
    print()
    
    # Create dataset for imputation
    data_genes = adata.raw.var_names.tolist()
    data_predict = pd.DataFrame(adata.raw.X.toarray(), columns = data_genes)
    sorted_val_dataset = pd.DataFrame(index = [i for i in range(0, len(adata.obs_names))])
    for column in features:
        if column in data_genes:
            sorted_val_dataset[column] = data_predict[column]
        else:
            sorted_val_dataset[column] = 0
    
    # Load dictionary of trained cell types
    proteins = pd.read_csv(os.path.join(path_model, 'proteins.csv'))
    proteins = list(proteins['feature_name'])
    proteins = [s + '_pred' for s in proteins]
    print('Successfully loaded list of features used for training model')
    print()
    
    # Load pretrained model
    loaded_model = TabNetRegressor()
    for file in os.listdir(path_model):
        if file.endswith('.zip'):
            loaded_model.load_model(os.path.join(path_model, file))
            print('Successfully loaded model')
            print()
    
    # Impute proteins
    predictions = loaded_model.predict(sorted_val_dataset.values)
    # Create DataFrame with imputed proteins
    predictions = pd.DataFrame(predictions, columns = proteins)
    
    # Create anndata object using imputed proteins
    adata_prot = anndata.AnnData(X = sparse.csc_matrix(predictions.values), 
                                 obs = adata.obs,
                                 var = pd.DataFrame(index = predictions.columns)
                                )

    if return_mdata == True:
        # Create mdata object using imputed proteins and rna from adata
        mdata = mu.MuData({rna_modality_name: adata, second_modality_name: adata_prot})
        return mdata
    else:
        return adata_prot


# Function for classification of imputed proteins based on Pearson correlation coeffecient
def high_corr(
    adata, 
    path_model = '',
    threshold = 0.6,
    subset = False
):
    '''
    Add adata.var['highly_correlated'] or subset adata.var considering the threshold based on the Pearson correlation coefficient.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix - output of scparadise.sceve.predict
    path_model : str, path object
        Path to the folder containing the trained scEve model.
    threshold : float, (default: 0.6)
        Pearson correlation coefficient threshold (0 - 1.0).     
    subset : bool, (default: False)
        If set to True, subset adata.var to proteins with correlation higher than the specified threshold.
        
    '''
    
    # load dataframe with correlations
    df_corr = pd.read_csv(os.path.join(path_model, 'df_corr.csv'), index_col=0)

    # Subset df_corr higher than the specified threshold
    df_corr_high = df_corr[df_corr['Pearson coef'] >= threshold].copy()
    df_corr_low = df_corr[df_corr['Pearson coef'] < threshold].copy()

    # Add adata.var['highly_correlated']
    adata.var['highly_correlated'] = pd.concat([pd.Series([True for x in range(len(df_corr_high.index))], index=df_corr_high.index),
                                                pd.Series([False for x in range(len(df_corr_low.index))], index=df_corr_low.index)])
    
    # Reduce the number of variables
    if subset:
        adata._inplace_subset_var(adata.var['highly_correlated'])


# Function to display available models in github
def available_models(
    
):
    '''
    Download dataframe with available trained scEve models.
    
    '''
    print("WARNING: RMSE, MASE, MSE are error metrics. Lower error metric value -> Better prediction.")
    print('RMSE - Root Mean Squared Error')
    print('MSE - Mean Squared Error')
    print('MAE - Mean Absolute Error')
    print()
    
    models = pd.read_csv('https://raw.githubusercontent.com/Chechekhins/scParadise/main/sceve_available_models.csv', sep=',')
    return models

# Function for downloading tuned pretrained models from github
def download_model(
    model_name = '',
    save_path = ''    
):
    '''
    Download pretrained tuned model for highly accurate cell surface protein abundance imputation.
    
    Parameters
    ----------
    model_name : str
        Name of the model from column 'model' from scparadise.sceve.available_models().
    save_path : str, path object
        Path to save the trained scEve model.
        
    '''

    # Create new directory with model
    save = os.path.join(save_path, model_name+'_scEve').replace("\\","/")
    if not os.path.exists(save):
        os.makedirs(save)

    # Download content of model
    fs = fsspec.filesystem("github", org="Chechekhins", repo="scParadise")
    fs.get(fs.ls(os.path.join('models_sceve', model_name+'_scEve').replace("\\","/")), save)
    
    