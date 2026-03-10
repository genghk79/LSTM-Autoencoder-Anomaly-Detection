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

class AEUtils():
    """Building, training, and inference class for LSTM autoencoder

    Args:
        input_size: Number of features in the input sequence.
        embed_size: Size of the hidden state embedding.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability for regularization.
        lr: Learning rate of optimiser.
        criterion: Loss function (e.g., MSELoss).
        optimiser: Optimizer for updating model parameters.
        optimiser_kwargs: Key word arguments for optimiser.
        curr_epoch: Current epoch number (for resuming training)
        train_loss: Training loss history (for resuming training)
        val_loss: Validation loss history (for resuming training)

    Attributes:
        ae_model: LSTM autoencoder model built.
        device: Device to run training on (CPU or CUDA).
        criterion: Loss function (e.g., MSELoss).
        optimiser: Optimizer for updating model parameters.
        optimiser_kwargs: Key word arguments for optimiser.
        curr_epoch: Current epoch number.
        train_loss: Training loss history.
        val_loss: Validation loss history.
    """
    def __init__(
            self,
            input_size: int,
            embed_size: int,
            num_layers: int,
            dropout: float,
            lr: float,
            criterion: nn.Module=nn.MSELoss(),
            optimiser: torch.optim.Optimizer=torch.optim.Adam,
            optimiser_kwargs: dict = None,
            curr_epoch: int=0,
            train_loss: list[float]=[],
            val_loss: list[float]=[],
    ):
        self.ae_model = LSTMAutoencoder(
            input_size=input_size, 
            embed_size=embed_size, 
            num_layers=num_layers, 
            dropout=dropout
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae_model = self.ae_model.to(self.device)

        self.criterion = criterion
        self.lr = lr
        optimiser_kwargs = optimiser_kwargs or {}
        self.optimiser = optimiser(self.ae_model.parameters(), lr=self.lr, **optimiser_kwargs)
        self.optimiser_kwargs = optimiser_kwargs
        self.curr_epoch = curr_epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        logger.info("Autoencoder and optimiser set up.")

    def train(self, dl_train: DataLoader, dl_val: DataLoader, num_epochs: int, plot_loss: bool) -> None:
        """Train the LSTM autoencoder model.

        Args:
            dl_train: DataLoader for training data.
            dl_val: DataLoader for validation data.
            num_epochs: Total number of epochs to train for.
            plot_loss: Whether to plot training and validation loss curves. Defaults to False.

        Raises:
            ValueError: If num_epochs is not greater than curr_epoch.
        """                 
        if num_epochs<=self.curr_epoch:
            raise ValueError(f"num_epochs={num_epochs} need to be greater than curr_epoch={self.curr_epoch}")

        start_time = time()
        pbar_epoch = tqdm(range(self.curr_epoch, num_epochs), 
                        desc=f"Training model for {num_epochs} epochs", unit='epoch')
        for epoch in pbar_epoch:
            
            self.ae_model.train()
            running_loss = 0.0

            # training loop
            for inputs in dl_train:
                
                data = inputs[0].to(self.device)
                self.optimiser.zero_grad()
                outputs = self.ae_model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item() * data.size(0)

            epoch_loss = running_loss / len(dl_train.dataset)
            self.train_loss.append(epoch_loss)

            self.ae_model.eval()
            running_loss = 0.0

            # validation loop
            with torch.no_grad():
                for inputs in dl_val:

                    data = inputs[0].to(self.device)
                    outputs = self.ae_model(data)
                    loss = self.criterion(outputs, data)

                    running_loss += loss.item() * data.size(0)

                epoch_loss = running_loss / len(dl_val.dataset)
                self.val_loss.append(epoch_loss)

            pbar_epoch.set_postfix({'Train Loss': f'{self.train_loss[-1]:.4f}', 'Validation Loss': f'{self.val_loss[-1]:.4f}'})

        end_time = time()
        logger.info(f"Autoencoder training epochs {self.curr_epoch} to {num_epochs} completed in {end_time-start_time:.0f}s.")

        # plot loss against epoch if requested
        if plot_loss:
            plt.plot(self.train_loss, label='Train Loss')
            plt.plot(self.val_loss, label='Validation Loss')      
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Autoencoder training progress') 
            plt.legend()
            plt.show()

    def inference(self, dl_test: DataLoader) -> torch.Tensor:
        """Perform inference using the trained autoencoder.

        Args:
            dl_test: DataLoader for test data.

        Returns:
            Reconstructed data tensor containing all batches.
        """

        self.ae_model.eval()
        recon_data = None

        # inference by batch
        pbar = tqdm(dl_test, desc=f"Autoencoder inferencing for {len(dl_test)} batches", unit='batch')
        with torch.no_grad():
            for inputs in pbar:
                data = inputs[0].to(self.device)
                outputs = self.ae_model(data)
                if recon_data is None:
                    recon_data = outputs.cpu()
                else:
                    recon_data = torch.cat([recon_data, outputs.cpu()])
        logger.info(f"Autoencoder inferencing for {len(dl_test)} test batches completed.")

        return recon_data


def save_ae(ae_class: AEUtils, scaler: StandardScaler, save_path: str) -> None:
    """Save autoencoder model and training state to disk.

    Args:
        ae_class: Trained autoencoder model to save.
        scaler: StandardScaler used for data normalization.
        save_path: Path where the model checkpoint will be saved.
    """

    # create directory to save model if it doesn't already exist
    path_obj = Path(save_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # save enough information for restarting training
    torch.save({
        'epoch': ae_class.curr_epoch,
        'model_state_dict': ae_class.ae_model.state_dict(),
        'input_size': ae_class.ae_model.input_size,
        'embed_size': ae_class.ae_model.hidden_size,
        'num_layers': ae_class.ae_model.num_layers,
        'dropout': ae_class.ae_model.dropout,
        'criterion': ae_class.criterion,
        'optimiser_state_dict': ae_class.optimiser.state_dict(),
        'optimiser_class': ae_class.optimiser.__class__,
        'optimiser_kwargs': ae_class.optimiser_kwargs,
        'lr': ae_class.optimiser.param_groups[0]['lr'],
        'train_loss': ae_class.train_loss,
        'val_loss': ae_class.val_loss,
        'scaler': scaler
    }, save_path)
    logger.info(f"Autoencoder saved in {save_path}.")
    

def load_ae(load_path: str) -> Tuple[AEUtils, StandardScaler]:
    """Load autoencoder model and training state from disk.

    Args:
        load_path: Path to the saved model checkpoint.

    Returns:
        Tuple containing:
            - Loaded autoencoder model class
            - StandardScaler used for data normalization
    """
    try:
        checkpoint = torch.load(load_path, weights_only=False)

        ae_class = AEUtils(
            input_size = checkpoint['input_size'],
            embed_size = checkpoint['embed_size'],
            num_layers = checkpoint['num_layers'],
            dropout = checkpoint['dropout'],
            lr = checkpoint['lr'],
            criterion = checkpoint['criterion'],
            optimiser = checkpoint['optimiser_class'],
            optimiser_kwargs = checkpoint['optimiser_kwargs'],
            curr_epoch = checkpoint['epoch'],
            train_loss = checkpoint['train_loss'],
            val_loss = checkpoint['val_loss'],
        )

        ae_class.ae_model.load_state_dict(checkpoint['model_state_dict'])
        ae_class.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        logger.info(f"Autoencoder loaded from {load_path}.")

        return ae_class, checkpoint['scaler']
    
    except Exception as e:
        logger.error(f"Unable to load from {load_path}: {e}")
