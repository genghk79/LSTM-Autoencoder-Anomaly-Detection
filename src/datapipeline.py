import pyreadr
import logging
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)


def create_data_tensor(X_df: pd.DataFrame, seq_len: int, 
                       scaler: Optional[StandardScaler] = None) -> Tuple[torch.Tensor, StandardScaler]:
    """Convert DataFrame to tensor sequences with standardization.

    Splits data into fixed-length sequences for time series modeling.
    Applies standardization using provided scaler or creates a new one.

    Args:
        X_df: DataFrame containing time series data with index columns.
        seq_len: Length of each sequence window.
        scaler: Optional StandardScaler for normalization. If None, creates and fits a new scaler.

    Returns:
        Tuple containing:
            - Tensor of shape (num_sequences, seq_len, num_features)
            - StandardScaler used for normalization

    Raises:
        ValueError: If original sequence length is not divisible by seq_len.
    """

    # seq_len is the window size for each data packet
    # data packets should not straddle two different simulation runs
    original_seq_length = X_df['sample'].max()
    if original_seq_length % seq_len != 0:
        raise ValueError(f"The original sequence length in data ({original_seq_length}) not divisible by seq_len {seq_len}.")
    
    X_tensor = torch.tensor([]) # initialise output tensor
    temp = X_df.iloc[:, 3:].values # get only the data feature parts of the df

    # standardise the data features
    if scaler==None:
        scaler = StandardScaler()    
        temp_scaled = scaler.fit_transform(temp)
    else:
        temp_scaled = scaler.transform(temp)

    # break data up into windows of seq_len, and stack windows into output tensor
    for i in range(0, X_df.shape[0]-seq_len+1, seq_len):
        seq = temp_scaled[i:i+seq_len]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, features)
        X_tensor = torch.cat((X_tensor, seq_tensor), dim=0)

    return X_tensor, scaler

def tensor_to_df(X_df: pd.DataFrame, X_tensor: torch.Tensor, scaler: StandardScaler) -> pd.DataFrame:
    """Convert tensor sequences back to DataFrame with inverse standardization.

    Args:
        X_df: Original DataFrame with index columns for reference.
        X_tensor: Tensor of shape (num_sequences, seq_len, num_features).
        scaler: StandardScaler to inverse transform the data.

    Returns:
        DataFrame with same structure as X_df containing reconstructed data.
    """

    # join windows in tensor back to make contiguous data
    for i in range(X_tensor.shape[0]):
        seq = X_tensor[i].detach().numpy()        
        if i == 0:
            reconstructed = seq
        else:
            reconstructed = np.vstack((reconstructed, seq))

    # undo standardisation
    reconstructed = scaler.inverse_transform(reconstructed)

    # make data into df with corresponding indexing columns
    df_reconstructed = X_df.copy()
    df_reconstructed.iloc[:, 3:] = reconstructed

    return df_reconstructed

def load_train_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, DataLoader, DataLoader, StandardScaler]:
    """Load and prepare training and validation data.

    Reads fault-free training data from RData file, splits into train/validation sets,
    creates tensors with standardization, and builds DataLoaders.

    Args:
        config: Configuration dictionary containing dataset parameters:
            - dataset.faultfree_train_path: Path to training RData file
            - dataset.faultfree_train_Robject: Name of R object in file
            - dataset.val_ratio: Validation split ratio
            - dataset.seq_len_train: Sequence length for training windows
            - dataset.batch_size: Batch size for DataLoaders

    Returns:
        Tuple containing:
            - Training DataFrame
            - Validation DataFrame
            - Training DataLoader
            - Validation DataLoader
            - StandardScaler fitted on training data
    """
    
    # load parameters from config
    train_path = config['dataset']['faultfree_train_path']
    train_Robject = config['dataset']['faultfree_train_Robject']
    val_ratio = config['dataset']['val_ratio']
    seq_len = config['dataset']['seq_len_train']
    batch_size = config['dataset']['batch_size']

    # read R data into dataframe
    print("Loading train data, this may take a while...")
    X_train_df = pyreadr.read_r(train_path)
    X_train_df = X_train_df[train_Robject]
    logger.info("Training RData loaded.")
    X_train_df['faultNumber'] = X_train_df['faultNumber'].astype(int)
    X_train_df['simulationRun'] = X_train_df['simulationRun'].astype(int)

    # split dataframe into train and validation sets
    train_val_split_run = np.floor(X_train_df['simulationRun'].max() * (1-val_ratio)).astype(int)
    X_val_df = X_train_df.loc[X_train_df['simulationRun']>train_val_split_run]
    X_train_df = X_train_df.loc[X_train_df['simulationRun']<=train_val_split_run]
    
    # extract data from dataframes as tensors
    X_train, scaler = create_data_tensor(X_train_df, seq_len=seq_len)
    X_val, _ = create_data_tensor(X_val_df, seq_len=seq_len, scaler=scaler)

    # create dataloaders
    ds_train = TensorDataset(X_train)
    ds_val = TensorDataset(X_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    logger.info("Training/validation dataframes and dataloaders returned.")
    logger.info(f"Training and validation dataframes sizes are {X_train_df.shape} & {X_val_df.shape}.")

    return X_train_df, X_val_df, dl_train, dl_val, scaler

def load_inference_data(config: dict, scaler: StandardScaler) -> Tuple[pd.DataFrame, DataLoader]:
    """Load and prepare test data for inference.

    Reads both fault-free and faulty test data from RData files, combines them,
    and creates DataLoader with standardization using provided scaler.

    Args:
        config: Configuration dictionary containing dataset parameters:
            - dataset.faultfree_test_path: Path to fault-free test RData file
            - dataset.faulty_test_path: Path to faulty test RData file
            - dataset.faultfree_test_Robject: Name of fault-free R object
            - dataset.faulty_test_Robject: Name of faulty R object
            - dataset.seq_len_test: Sequence length for test windows
            - dataset.batch_size: Batch size for DataLoader
            - dataset.test_runs_to_load: Number of simulation runs to load per fault
        scaler: StandardScaler fitted on training data for normalization.

    Returns:
        Tuple containing:
            - Test DataFrame combining fault-free and faulty data
            - Test DataLoader

    Raises:
        ValueError: If test_runs_to_load exceeds 500.
    """

    # load parameters from config
    faultfree_test_path = config['dataset']['faultfree_test_path']
    faulty_test_path = config['dataset']['faulty_test_path']
    faultfree_test_Robject = config['dataset']['faultfree_test_Robject']
    faulty_test_Robject = config['dataset']['faulty_test_Robject']
    seq_len = config['dataset']['seq_len_test']
    batch_size = config['dataset']['batch_size']
    testruns2load = config['dataset']['test_runs_to_load']

    if testruns2load > 500:
        raise ValueError("test_runs_to_load exceed number of simulationRun in data")

    # read R data into dataframe
    print("Loading test data, this may take a while...")
    X_test_faultfree_df = pyreadr.read_r(faultfree_test_path)
    X_test_faultfree_df = X_test_faultfree_df[faultfree_test_Robject]
    logger.info("Faultfree test RData loaded.")
    X_test_faulty_df = pyreadr.read_r(faulty_test_path)
    X_test_faulty_df = X_test_faulty_df[faulty_test_Robject]
    logger.info("Faulty test RData loaded.")
    X_test_faultfree_df['faultNumber'] = X_test_faultfree_df['faultNumber'].astype(int)
    X_test_faulty_df['faultNumber'] = X_test_faulty_df['faultNumber'].astype(int)
    X_test_faultfree_df['simulationRun'] = X_test_faultfree_df['simulationRun'].astype(int)
    X_test_faulty_df['simulationRun'] = X_test_faulty_df['simulationRun'].astype(int)

    # combine just the desired number of simulationRuns into a single test dataframe
    X_test_df = pd.concat([X_test_faultfree_df.loc[X_test_faultfree_df['simulationRun']<=testruns2load], 
                               X_test_faulty_df.loc[X_test_faulty_df['simulationRun']<=testruns2load]])

    # extract data from dataframes as tensors
    X_tensor, _ = create_data_tensor(X_test_df, seq_len=seq_len, scaler=scaler)

    # create dataloaders
    ds_test = TensorDataset(X_tensor)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    logger.info(f"Test dataframe and dataloader returned, {testruns2load} runs for each faultNumber.")
    logger.info(f"Test dataframe size is {X_test_df.shape}.")

    return X_test_df, dl_test