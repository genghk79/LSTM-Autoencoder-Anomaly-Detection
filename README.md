# LSTM Autoencoder-based Anomaly Detection

___
## Introduction
This project is a personal practice attempt at performing anomaly detection on a multivariate time-series dataset. The dataset chosen is the Tennessee Eastman dataset, and anomalies in the data are detected through the reconstruction errors of an LSTM autoencoder.

In this approach, the autoencoder is first trained to reproduce the data of a 'normal' dataset. The assumption is that when anomalous data is later presented to the autoencoder, it would struggle to reproduce them and therefore higher reproduction/reconstruction errors would be incurred.
___
## The Tennessee Eastman Dataset
Source: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
- Download the dataset, and make sure the following four files are in the `./data/` directory
    - `TEP_FaultFree_Testing.RData`
    - `TEP_FaultFree_Training.RData`
    - `TEP_Faulty_Testing.RData`
    - `TEP_Faulty_Training.RData`

- More insight about the Tennessee Eastman Dataset can be found here:
    - https://keepfloyding.github.io/posts/Ten-East-Proc-Intro/
    - https://keepfloyding.github.io/posts/data-explor-TEP-1/
    - https://keepfloyding.github.io/posts/data-explor-TEP-2/
    - https://keepfloyding.github.io/posts/data-explor-TEP-3/


___
## Overview of Code
The essential code are found in the `src` directory.
- `datapipeline.py`: Functions for loading the Tennessee Eastman dataset and preparing the data for the autoencoder. Key parameters are loaded from `./configs/config.yaml`.
- `autoencoder.py`: Definitions of the encoder, decoder and autoencoder classes
- `ae_utils.py`: Functions for building, training, saving, loading, and inferencing from the autoencoder.
- `eval_tools.py`: Definition of an evaluator class specially designed for the autoencoder's anomaly detection performance on the Tennessee Eastman dataset.

Example of how the code can be used for anomaly detection is shown in the following section.

If adapting the LSTM autoencoder for other datasets or for other purpose, `autoencoder.py` and `ae_utils.py` can probably be reused with minimal modification. 

___
## Anomaly Detection Process
This process detects anomalies based on the reconstruction errors of the autoencoder (AE). Since the dataset is a time-series dataset, the AE will be a LSTM AE.

We will be training the AE with just the fault-free training dataset to establish what normal data should look like.

The AE will then be tested through the reconstruction of data from both the fault-free and faulty testing datasets.

### 1. Load Training Data


```python
from src.datapipeline import load_train_data
from src.general_utils import load_config

config = load_config('./configs/config.yaml')
X_train_df, X_val_df, dl_train, dl_val, scaler = load_train_data(config)
```

    INFO - Configuration loaded from ./configs/config.yaml


    Loading train data, this may take a while...


    INFO - Training RData loaded.
    INFO - Training/validation dataframes and dataloaders returned.
    INFO - Training and validation dataframes sizes are (225000, 55) & (25000, 55).


### 2. Set Up AE 


```python
from src.ae_utils import AEUtils

# each window shape is (batch_size, seq_len, num_features)
input_size = next(iter(dl_train))[0].shape[2]

ae_model = AEUtils(input_size=input_size,
                   embed_size=config['ae_params']['embed_size'],
                   num_layers=config['ae_params']['num_layers'],
                   dropout=config['ae_params']['dropout'],
                   lr=config["train_params"]['lr'])
```

    INFO - Autoencoder and optimiser set up.


### 3. Train AE


```python
ae_model.train(dl_train=dl_train, dl_val=dl_val, num_epochs=config['train_params']['num_epochs'], plot_loss=True)
```

    Training model for 3000 epochs: 100%|██████████| 3000/3000 [39:16<00:00,  1.27epoch/s, Train Loss=0.6818, Validation Loss=0.6972]
    INFO - Autoencoder training epochs 0 to 3000 completed in 2357s.



    
![png](AE_AD_files/AE_AD_6_1.png)
    



```python
from src.ae_utils import save_ae

save_path = config['save_path']

save_ae(ae_model, scaler, save_path)
```

    INFO - Autoencoder saved in ./models/ae_model_saved.pth.


### 3.1. Load AE & Resume Training

The `save_ae`, `load_ae` and `train_ae` functions allow for training to be resumed from the saved data.


```python
from src.ae_utils import load_ae

ae_model, scaler = load_ae(save_path)

# resume training for an additional 1000 epochs
num_epochs = ae_model.curr_epoch + 1000

ae_model.train(dl_train=dl_train, dl_val=dl_val, num_epochs=num_epochs, plot_loss=True)
```

    INFO - Autoencoder and optimiser set up.
    INFO - Autoencoder loaded from ./models/ae_model_saved.pth.
    Training model for 1000 epochs: 100%|██████████| 1000/1000 [12:58<00:00,  1.29epoch/s, Train Loss=0.6724, Validation Loss=0.6957]
    INFO - Autoencoder training epochs 0 to 1000 completed in 778s.



    
![png](AE_AD_files/AE_AD_9_1.png)
    



```python
save_ae(ae_model, scaler, save_path)
```

    INFO - Autoencoder saved in ./models/ae_model_saved.pth.


### 4. Data Reconstruction and Anomaly Detection with AE
The Tennessee Eastman test dataset has fault-free as well as faulty data to test with.

### 4.1 Load Test Data
There's the fault-free data, and faults no. 1-20. Each set has 500 simulation runs. We will only load 20 runs of each for testing.


```python
from src.datapipeline import load_inference_data

# modify config.yaml to change number of runs loaded
X_test_df, dl_test = load_inference_data(config, scaler)
```

    Loading test data, this may take a while...


    INFO - Faultfree test RData loaded.
    INFO - Faulty test RData loaded.
    INFO - Test dataframe and dataloader returned, 20 runs for each faultNumber.
    INFO - Test dataframe size is (403200, 55).


### 4.2 Process Test Data for Evaluation
Creating the `AE_eval` class object will trigger an inference run on the given test data.


```python
from src.eval_tools import AE_eval

ae_eval = AE_eval(X_test_df, dl_test, ae_model, scaler)
```

    Autoencoder inferencing for 53 batches: 100%|██████████| 53/53 [00:02<00:00, 24.76batch/s]
    INFO - Autoencoder inferencing for 53 test batches completed.


Calling the `reconstruction_RMSE()` method populates a dataframe with the RMSE of all test simulation runs.


```python
ae_eval.reconstruction_RMSE()
```

    faultNumber: 100%|██████████| 21/21 [00:58<00:00,  2.78s/it]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>faultNumber</th>
      <th>simulationRun</th>
      <th>xmeas_1</th>
      <th>xmeas_2</th>
      <th>xmeas_3</th>
      <th>xmeas_4</th>
      <th>xmeas_5</th>
      <th>xmeas_6</th>
      <th>xmeas_7</th>
      <th>xmeas_8</th>
      <th>...</th>
      <th>xmv_2</th>
      <th>xmv_3</th>
      <th>xmv_4</th>
      <th>xmv_5</th>
      <th>xmv_6</th>
      <th>xmv_7</th>
      <th>xmv_8</th>
      <th>xmv_9</th>
      <th>xmv_10</th>
      <th>xmv_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.880089</td>
      <td>1.003087</td>
      <td>0.925694</td>
      <td>0.907644</td>
      <td>1.017489</td>
      <td>1.016953</td>
      <td>0.426612</td>
      <td>0.959295</td>
      <td>...</td>
      <td>0.898315</td>
      <td>0.877591</td>
      <td>1.006192</td>
      <td>0.590311</td>
      <td>0.881636</td>
      <td>1.024897</td>
      <td>1.000829</td>
      <td>0.264655</td>
      <td>0.971868</td>
      <td>1.041640</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.791371</td>
      <td>0.999688</td>
      <td>0.850293</td>
      <td>0.854301</td>
      <td>0.996791</td>
      <td>1.012993</td>
      <td>0.373188</td>
      <td>0.916612</td>
      <td>...</td>
      <td>0.835365</td>
      <td>0.788064</td>
      <td>0.957395</td>
      <td>0.502607</td>
      <td>0.877210</td>
      <td>1.030475</td>
      <td>0.962912</td>
      <td>0.190672</td>
      <td>0.983016</td>
      <td>1.008679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.792839</td>
      <td>1.085337</td>
      <td>0.868967</td>
      <td>0.867859</td>
      <td>0.970959</td>
      <td>0.983802</td>
      <td>0.459200</td>
      <td>0.928987</td>
      <td>...</td>
      <td>0.884623</td>
      <td>0.795389</td>
      <td>1.032016</td>
      <td>0.571718</td>
      <td>0.879546</td>
      <td>0.977487</td>
      <td>1.000270</td>
      <td>0.297306</td>
      <td>0.989136</td>
      <td>1.026241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.826583</td>
      <td>1.065298</td>
      <td>0.880445</td>
      <td>0.865911</td>
      <td>1.009061</td>
      <td>1.021981</td>
      <td>0.439997</td>
      <td>0.905820</td>
      <td>...</td>
      <td>0.885976</td>
      <td>0.828836</td>
      <td>0.968427</td>
      <td>0.507255</td>
      <td>0.832542</td>
      <td>0.998083</td>
      <td>0.961719</td>
      <td>0.287971</td>
      <td>0.981225</td>
      <td>0.969403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.821180</td>
      <td>0.986490</td>
      <td>0.882254</td>
      <td>0.908796</td>
      <td>0.987980</td>
      <td>1.001493</td>
      <td>0.379872</td>
      <td>0.928540</td>
      <td>...</td>
      <td>0.876636</td>
      <td>0.820239</td>
      <td>0.946197</td>
      <td>0.503354</td>
      <td>0.893168</td>
      <td>1.022826</td>
      <td>1.002401</td>
      <td>0.218615</td>
      <td>0.954146</td>
      <td>0.977718</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>415</th>
      <td>20.0</td>
      <td>16.0</td>
      <td>0.913361</td>
      <td>1.052760</td>
      <td>0.957471</td>
      <td>0.954048</td>
      <td>0.993662</td>
      <td>1.041146</td>
      <td>0.931407</td>
      <td>0.995391</td>
      <td>...</td>
      <td>0.972520</td>
      <td>0.909973</td>
      <td>0.984693</td>
      <td>4.769154</td>
      <td>1.012694</td>
      <td>1.000100</td>
      <td>1.023412</td>
      <td>0.569944</td>
      <td>1.064683</td>
      <td>1.002294</td>
    </tr>
    <tr>
      <th>416</th>
      <td>20.0</td>
      <td>17.0</td>
      <td>0.900638</td>
      <td>1.098103</td>
      <td>0.964945</td>
      <td>0.916011</td>
      <td>1.031741</td>
      <td>1.004140</td>
      <td>0.873783</td>
      <td>0.983868</td>
      <td>...</td>
      <td>0.926652</td>
      <td>0.897734</td>
      <td>0.976432</td>
      <td>4.786109</td>
      <td>0.951849</td>
      <td>0.977040</td>
      <td>1.027259</td>
      <td>0.476004</td>
      <td>1.056296</td>
      <td>1.026162</td>
    </tr>
    <tr>
      <th>417</th>
      <td>20.0</td>
      <td>18.0</td>
      <td>0.911694</td>
      <td>1.114207</td>
      <td>0.941790</td>
      <td>0.949755</td>
      <td>1.013030</td>
      <td>1.049410</td>
      <td>0.983578</td>
      <td>0.977883</td>
      <td>...</td>
      <td>0.958664</td>
      <td>0.911338</td>
      <td>0.983964</td>
      <td>4.657758</td>
      <td>0.972685</td>
      <td>0.990367</td>
      <td>1.017721</td>
      <td>0.529892</td>
      <td>1.094599</td>
      <td>1.022070</td>
    </tr>
    <tr>
      <th>418</th>
      <td>20.0</td>
      <td>19.0</td>
      <td>0.911375</td>
      <td>1.090411</td>
      <td>0.927493</td>
      <td>0.928917</td>
      <td>1.002062</td>
      <td>1.074637</td>
      <td>0.890171</td>
      <td>0.958829</td>
      <td>...</td>
      <td>0.938975</td>
      <td>0.913829</td>
      <td>0.984918</td>
      <td>4.752060</td>
      <td>0.948111</td>
      <td>1.012444</td>
      <td>1.044894</td>
      <td>0.607917</td>
      <td>1.127797</td>
      <td>1.013158</td>
    </tr>
    <tr>
      <th>419</th>
      <td>20.0</td>
      <td>20.0</td>
      <td>0.989460</td>
      <td>1.185355</td>
      <td>0.994532</td>
      <td>1.018454</td>
      <td>1.003444</td>
      <td>1.041711</td>
      <td>0.948219</td>
      <td>0.995981</td>
      <td>...</td>
      <td>0.991665</td>
      <td>0.990674</td>
      <td>1.028022</td>
      <td>4.927338</td>
      <td>0.949200</td>
      <td>0.966408</td>
      <td>0.983996</td>
      <td>0.744534</td>
      <td>1.142989</td>
      <td>1.031503</td>
    </tr>
  </tbody>
</table>
<p>420 rows × 54 columns</p>
</div>



Verify that the fault-free test set has low RMSE for all features. 

The RMSE is calculated on standardised feature values. So we would expect the RMSE to be around 1 or lower.


```python
ae_eval.plot_RMSE_distributions(faultNumber=0)
```


    
![png](AE_AD_files/AE_AD_19_0.png)
    


Verify that the reconstructed data follows the general pattern of the test data.

The semi-transparent area show the min-max of the 20 runs.


```python
ae_eval.plot_feature(feature='xmeas_1', faultNumber=0)
```


    
![png](AE_AD_files/AE_AD_21_0.png)
    


For each `faultNumber`, which features have RMSE that exceed a certain threshold value?


```python
ae_eval.find_high_RMSE(RMSE_threshold=1.4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>faultNumber</th>
      <th>highMSEcolumns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[xmeas_1, xmeas_3, xmeas_4, xmeas_7, xmeas_8, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[xmeas_3, xmeas_4, xmeas_6, xmeas_7, xmeas_10,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[xmv_10]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>[xmeas_20, xmv_11]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>[xmeas_1, xmeas_2, xmeas_3, xmeas_4, xmeas_7, ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>[xmeas_1, xmeas_3, xmeas_4, xmeas_7, xmeas_8, ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>[xmeas_1, xmeas_3, xmeas_4, xmeas_7, xmeas_8, ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>[xmeas_18]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>[xmeas_9, xmv_10]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>[xmeas_1, xmeas_2, xmeas_3, xmeas_4, xmeas_6, ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>[xmeas_1, xmeas_2, xmeas_3, xmeas_4, xmeas_7, ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>[xmeas_9, xmeas_21, xmv_10]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>[xmeas_9, xmeas_11, xmeas_18, xmeas_19, xmeas_...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>[xmeas_1, xmeas_2, xmeas_3, xmeas_4, xmeas_6, ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>[xmeas_5]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>[xmeas_11, xmeas_13, xmeas_20, xmeas_22, xmv_5]</td>
    </tr>
  </tbody>
</table>
</div>



Visually inspect an example to see what the RMSE across features look like.


```python
ae_eval.plot_RMSE_distributions(faultNumber=4)
```


    
![png](AE_AD_files/AE_AD_25_0.png)
    


Visually inspect the time-series to see how the data is anomalous. Compare that with the fault-free case.


```python
feature = 'xmv_10'

ae_eval.plot_feature(feature=feature, faultNumber=4, simulationRun=1)
ae_eval.plot_feature(feature=feature, faultNumber=0, simulationRun=1)
```


    
![png](AE_AD_files/AE_AD_27_0.png)
    



    
![png](AE_AD_files/AE_AD_27_1.png)
    


___
## Limitations
It can be seen that certain faulty datasets (e.g. `faultNumber=3`) did not have any features that were flagged as having a high RMSE, and thus would not be detected as anomalous. A closer inspection of the original and reconstructed data would be necessary to understand where the approach fails. 

Without further analysis, one possible failure mode that we can easily imagine is if the faulty data is characterised by an *absence* of noise or variations. The reconstruction RMSE would more likely be lower rather than higher than in the fault-free case, and escape detection in this approach.

From the feature plots, we can tell that the autoencoder is unable to reproduce all the noise and random high frequency variations even in the fault-free data, and behaves somewhat like a low-pass filter. This is not unexpected as the embeddings would only retain the salient features in the data, and this should not include random fluctuations. But this also prevents us from detecting anomalies that produce lower reproduction RMSE values via this approach.


