import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder for autoencoder architecture.

    Encodes input sequences into a fixed-size hidden representation using LSTM layers.

    Args:
        input_size: Number of features in the input sequence.
        embed_size: Size of the hidden state embedding.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability for regularization.
    """

    def __init__(self, input_size: int, embed_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            num_layers=self.num_layers, dropout=dropout,
                            batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Hidden state tensor of shape (num_layers, batch_size, hidden_size).
        """
        _, (hidden, _) = self.lstm(x)
        return hidden
    
class LSTMDecoder(nn.Module):
    """LSTM decoder for autoencoder architecture.

    Decodes hidden representations back into output sequences using LSTM layers.

    Args:
        embed_size: Size of the hidden state embedding.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability for regularization.
        output_size: Number of features in the output sequence.
    """

    def __init__(self, embed_size: int, num_layers: int, dropout: float, output_size: int) -> None:
        super().__init__()
        self.hidden_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            num_layers=self.num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size).
        """
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    
class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence reconstruction.

    Combines an LSTM encoder and decoder to compress and reconstruct input sequences.
    Used for anomaly detection by measuring reconstruction error.

    Args:
        input_size: Number of features in the input sequence.
        embed_size: Size of the hidden state embedding (compression size).
        num_layers: Number of stacked LSTM layers in encoder and decoder.
        dropout: Dropout probability for regularization.
    """

    def __init__(self, input_size: int, embed_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = LSTMEncoder(self.input_size, self.hidden_size,
                                   self.num_layers, self.dropout)
        self.decoder = LSTMDecoder(self.hidden_size, self.num_layers, self.dropout, self.input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, input_size).
        """
        seq_len = x.shape[1]
        encoded_x = self.encoder(x)
        encoded_x = encoded_x[-1].squeeze()
        encoded_x = encoded_x.unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x