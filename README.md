# LSTM Autoencoder for Time-Series Anomaly Detection

An unsupervised anomaly detection system using an LSTM autoencoder. The autoencoder is trained exclusively on normal (fault-free) data so it learns to reconstruct normal patterns. At inference time, anomalous data produces high reconstruction error (RMSE), signalling a potential anomaly.

For a full worked example on the Tennessee Eastman Process dataset, including analysis of results and discussion of limitations, see [**AE_AD.ipynb**](AE_AD.ipynb).

## Project Structure

```
LSTM_AE_AD/
├── AE_AD.ipynb              # End-to-end example notebook (Tennessee Eastman dataset)
├── configs/
│   ├── config.yaml          # Hyperparameters, data paths, training settings
│   └── logging.yaml         # Logging configuration
├── src/
│   ├── autoencoder.py       # LSTM Encoder, Decoder, and Autoencoder model definitions
│   ├── ae_utils.py          # AEUtils class for building, training, saving/loading, inference
│   ├── datapipeline.py      # Data loading, windowing, scaling, DataLoader creation
│   ├── eval_tools.py        # AE_eval class for RMSE computation, plotting, anomaly reporting
│   └── general_utils.py     # Config and logging setup utilities
├── data/                    # Dataset files (not tracked by git)
├── models/                  # Saved model checkpoints (not tracked by git)
├── requirements.txt
└── LICENSE
```

## Setup

### Requirements

Python 3.8+ with the following packages:

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn seaborn pyreadr pyyaml tqdm python-json-logger
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Configuration

All hyperparameters and data paths are set in `configs/config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `embed_size` | 100 | LSTM hidden state size (bottleneck dimension) |
| `num_layers` | 2 | Stacked LSTM layers in encoder and decoder |
| `dropout` | 0.2 | Dropout between LSTM layers |
| `lr` | 0.0001 | Adam learning rate |
| `num_epochs` | 3000 | Training epochs |
| `seq_len_train` | 125 | Window length for training data |
| `seq_len_test` | 120 | Window length for test data |
| `batch_size` | 64 | Training batch size |
| `val_ratio` | 0.1 | Fraction of runs held out for validation |

## How It Works

1. **Train** the autoencoder on fault-free data only. The model learns to reconstruct normal time-series patterns.
2. **Run inference** on unseen data (both normal and potentially anomalous).
3. **Compute per-feature RMSE** between original and reconstructed signals for each simulation run.
4. **Flag anomalies** where RMSE exceeds a chosen threshold.

### Model Architecture

The autoencoder compresses a multivariate time-series window into a fixed-size hidden state (the bottleneck), then reconstructs the original window from that compressed representation.

- **Encoder:** Stacked LSTM that reads the input sequence and produces a final hidden state.
- **Bottleneck:** The last LSTM layer's hidden state, repeated across the sequence length.
- **Decoder:** Stacked LSTM followed by a linear projection back to the original feature dimensions.

## Preparing Your Own Time-Series Data

To use this code with your own dataset, your data must be structured as a pandas DataFrame with the following columns:

| Column | Type | Description |
|---|---|---|
| `faultNumber` | int | Label for the condition. Use `0` for normal/fault-free data. Use other integers (1, 2, 3, ...) for different anomaly types. |
| `simulationRun` | int | Identifies independent runs or sequences. Each run is treated as a separate time series. |
| `sample` | int | Time step index within a run (1-indexed, consecutive). |
| feature columns | float | One column per sensor/variable. These are the values the autoencoder will learn to reconstruct. |

**Important constraints:**

- **`seq_len` must evenly divide the number of samples per run.** The data is sliced into non-overlapping windows of `seq_len` time steps. Windows never cross run boundaries. For example, if each run has 500 samples, valid `seq_len` values include 25, 50, 100, 125, 250, 500.
- **Training data should contain only normal (fault-free) data** (`faultNumber = 0`). The autoencoder learns what "normal" looks like, so anomalies in the training set will degrade detection performance.
- **The same scaler must be used for training and inference.** The pipeline fits a `StandardScaler` on the training data and reuses it for all subsequent data. This is handled automatically by the `datapipeline` module.

### Adapting the Data Pipeline

If your data is not in `.RData` format, you will need to modify the `load_train_data()` and `load_inference_data()` functions in `src/datapipeline.py` to load your data into the DataFrame format described above. The core windowing function `create_data_tensor()` and the rest of the pipeline will work without changes as long as the DataFrame schema is correct.

## Example: Tennessee Eastman Process

The notebook [**AE_AD.ipynb**](AE_AD.ipynb) walks through the full pipeline using the Tennessee Eastman Process (TEP) simulation dataset, a widely-used benchmark for industrial process monitoring with 52 sensor/actuator features and 20 distinct fault types. The notebook covers:

- Loading and preparing the data
- Training the LSTM autoencoder (3000 epochs, ~39 minutes on GPU)
- Saving and resuming training from checkpoints
- Running inference on test data containing both normal and faulty runs
- Computing and visualising per-feature RMSE distributions
- Identifying which faults are detected and which are missed
- Investigating why certain fault types evade detection

## License

MIT License. See [LICENSE](LICENSE) for details.
