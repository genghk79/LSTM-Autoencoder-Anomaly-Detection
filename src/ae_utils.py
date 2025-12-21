import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from src.autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)


def train_ae(ae_model: LSTMAutoencoder, dl_train: DataLoader, dl_val: DataLoader,
             criterion: nn.Module, optimiser: torch.optim.Optimizer, num_epochs: int,
             device: torch.device, curr_epoch: int = 0, train_loss: list[float] = [],
             val_loss: list[float] = [], plot_loss: bool = False) -> Tuple[list[float], 
                                                                           list[float], 
                                                                           LSTMAutoencoder, 
                                                                           torch.optim.Optimizer]:
    """Train the LSTM autoencoder model.

    Args:
        ae_model: LSTM autoencoder model to train.
        dl_train: DataLoader for training data.
        dl_val: DataLoader for validation data.
        criterion: Loss function (e.g., MSELoss).
        optimiser: Optimizer for updating model parameters.
        num_epochs: Total number of epochs to train for.
        device: Device to run training on (CPU or CUDA).
        curr_epoch: Current epoch number for resuming training. Defaults to 0.
        train_loss: List of training losses from previous epochs. Defaults to [].
        val_loss: List of validation losses from previous epochs. Defaults to [].
        plot_loss: Whether to plot training and validation loss curves. Defaults to False.

    Returns:
        Tuple containing:
            - Updated training loss list
            - Updated validation loss list
            - Trained autoencoder model
            - Updated optimizer

    Raises:
        ValueError: If num_epochs is not greater than curr_epoch.
    """

    if num_epochs<=curr_epoch:
        raise ValueError(f"num_epochs={num_epochs} need to be greater than curr_epoch={curr_epoch}")

    start_time = time()
    pbar_epoch = tqdm(range(curr_epoch, num_epochs), 
                      desc=f"Training model for {num_epochs} epochs", unit='epoch')
    for epoch in pbar_epoch:
        
        ae_model.train()
        running_loss = 0.0

        # training loop
        for inputs in dl_train:
            
            data = inputs[0].to(device)
            optimiser.zero_grad()
            outputs = ae_model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dl_train.dataset)
        train_loss.append(epoch_loss)

        ae_model.eval()
        running_loss = 0.0

        # validation loop
        with torch.no_grad():
            for inputs in dl_val:

                data = inputs[0].to(device)
                outputs = ae_model(data)
                loss = criterion(outputs, data)

                running_loss += loss.item() * data.size(0)

            epoch_loss = running_loss / len(dl_val.dataset)
            val_loss.append(epoch_loss)

        pbar_epoch.set_postfix({'Train Loss': f'{train_loss[-1]:.4f}', 'Validation Loss': f'{val_loss[-1]:.4f}'})

    end_time = time()
    logger.info(f"Autoencoder training epochs {curr_epoch} to {num_epochs} completed in {end_time-start_time:.0f}s.")

    # plot loss against epoch if requested
    if plot_loss:
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')      
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder training progress') 
        plt.legend()
        plt.show()

    return train_loss, val_loss, ae_model, optimiser

def inference_ae(ae_model: LSTMAutoencoder, dl_test: DataLoader, device: torch.device) -> torch.Tensor:
    """Perform inference using the trained autoencoder.

    Args:
        ae_model: Trained LSTM autoencoder model.
        dl_test: DataLoader for test data.
        device: Device to run inference on (CPU or CUDA).

    Returns:
        Reconstructed data tensor containing all batches.
    """

    ae_model.eval()
    recon_data = None

    # inference by batch
    pbar = tqdm(dl_test, desc=f"Autoencoder inferencing for {len(dl_test)} batches", unit='batch')
    with torch.no_grad():
        for inputs in pbar:
            data = inputs[0].to(device)
            outputs = ae_model(data)
            if recon_data is None:
                recon_data = outputs.cpu()
            else:
                recon_data = torch.cat([recon_data, outputs.cpu()])
    logger.info(f"Autoencoder inferencing for {len(dl_test)} test batches completed.")

    return recon_data

def build_ae(input_size: int, embed_size: int, 
             num_layers: int, dropout: float, lr: float) -> Tuple[LSTMAutoencoder, torch.device, 
                                                                  nn.Module, torch.optim.Optimizer]:
    """Build and initialize the LSTM autoencoder with optimizer and loss criterion.

    Args:
        input_size: Number of features in the input sequence.
        embed_size: Size of the hidden state embedding.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability for regularization.
        lr: Learning rate for the optimizer.

    Returns:
        Tuple containing:
            - Initialized autoencoder model
            - Device (CPU or CUDA)
            - Loss criterion (MSELoss)
            - Optimizer (Adam)
    """

    ae_model = LSTMAutoencoder(input_size=input_size, embed_size=embed_size, num_layers=num_layers, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = ae_model.to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(ae_model.parameters(), lr=lr)
    logger.info("Autoencoder and optimiser set up.")

    return ae_model, device, criterion, optimiser

def save_ae(ae_model: LSTMAutoencoder, scaler: StandardScaler, optimiser: torch.optim.Optimizer,
            num_epochs: int, train_loss: list[float], val_loss: list[float], save_path: str) -> None:
    """Save autoencoder model and training state to disk.

    Args:
        ae_model: Trained autoencoder model to save.
        scaler: StandardScaler used for data normalization.
        optimiser: Optimizer used during training.
        num_epochs: Number of epochs completed.
        train_loss: List of training losses per epoch.
        val_loss: List of validation losses per epoch.
        save_path: Path where the model checkpoint will be saved.
    """

    # create directory to save model if it doesn't already exist
    path_obj = Path(save_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # save enough information for restarting training
    torch.save({
            'epoch': num_epochs,
            'model_state_dict': ae_model.state_dict(),
            'input_size': ae_model.input_size,
            'embed_size': ae_model.hidden_size,
            'num_layers': ae_model.num_layers,
            'dropout': ae_model.dropout,
            'optimiser_state_dict': optimiser.state_dict(),
            'optimiser_class': optimiser.__class__,
            'lr': optimiser.param_groups[0]['lr'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'scaler': scaler
            }, save_path)
    logger.info(f"Autoencoder saved in {save_path}.")
    
def load_ae(load_path: str) -> Tuple[LSTMAutoencoder, torch.optim.Optimizer, torch.device, 
                                     StandardScaler, int, list[float], list[float]]:
    """Load autoencoder model and training state from disk.

    Args:
        load_path: Path to the saved model checkpoint.

    Returns:
        Tuple containing:
            - Loaded autoencoder model
            - Loaded optimizer
            - Device (CPU or CUDA)
            - StandardScaler used for data normalization
            - Number of epochs completed
            - Training loss history
            - Validation loss history
    """
    try:
        checkpoint = torch.load(load_path, weights_only=False)

        ae_model = LSTMAutoencoder(input_size=checkpoint['input_size'], embed_size=checkpoint['embed_size'], 
                                num_layers=checkpoint['num_layers'], dropout=checkpoint['dropout'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_model = ae_model.to(device)

        optimiser = checkpoint['optimiser_class'](ae_model.parameters(), lr=checkpoint['lr'])

        ae_model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        logger.info(f"Autoencoder loaded from {load_path}.")

        return ae_model, optimiser, device, checkpoint['scaler'], checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    
    except Exception as e:
        logger.error(f"Unable to load from {load_path}: {e}")
