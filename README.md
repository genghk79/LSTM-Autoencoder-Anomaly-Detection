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

This `README.md` is generated from the `AE_AD.ipynb` notebook.

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

    {"asctime": "2025-12-20T00:34:49+0800", "process": 19692, "name": "src.general_utils", "levelname": "INFO", "message": "Configuration loaded from ./configs/config.yaml"}


    Loading train data, this may take a while...


    {"asctime": "2025-12-20T00:34:52+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Training RData loaded."}
    {"asctime": "2025-12-20T00:35:00+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Training/validation dataframes and dataloaders returned."}
    {"asctime": "2025-12-20T00:35:00+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Training and validation dataframes sizes are (225000, 55) & (25000, 55)."}


### 2. Set Up AE 


```python
from src.ae_utils import build_ae

# each window shape is (batch_size, seq_len, num_features)
input_size = next(iter(dl_train))[0].shape[2]

ae_model, device, criterion, optimiser = build_ae(input_size=input_size,
                                                  embed_size=config['ae_params']['embed_size'],
                                                  num_layers=config['ae_params']['num_layers'],
                                                  dropout=config['ae_params']['dropout'],
                                                  lr=config['train_params']['lr'])
```

    {"asctime": "2025-12-20T00:35:07+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder and optimiser set up."}


### 3. Train AE


```python
from src.ae_utils import train_ae

num_epochs = config['train_params']['num_epochs']

train_loss, val_loss, ae_model, optimiser = train_ae(ae_model=ae_model, 
                                                      dl_train=dl_train, 
                                                      dl_val=dl_val, 
                                                      criterion=criterion, 
                                                      optimiser=optimiser, 
                                                      num_epochs=num_epochs, 
                                                      device=device,
                                                      plot_loss=True)
```

    Training model for 3000 epochs: 100%|██████████| 3000/3000 [39:42<00:00,  1.26epoch/s, Train Loss=0.6833, Validation Loss=0.6923]
    {"asctime": "2025-12-20T01:14:50+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder training epochs 0 to 3000 completed in 2383s."}



    
![png](AE_AD_files/AE_AD_6_1.png)
    



```python
from src.ae_utils import save_ae

save_path = config['save_path']

save_ae(ae_model, scaler, optimiser, num_epochs, train_loss, val_loss, save_path)
```

    {"asctime": "2025-12-20T01:14:51+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder saved in ./models/ae_model_saved.pth."}


### 3.1. Load AE & Resume Training

The `save_ae`, `load_ae` and `train_ae` functions allow for training to be resumed from the saved data.


```python
from src.ae_utils import load_ae

ae_model, optimiser, device, scaler, curr_epoch, train_loss, val_loss = load_ae(save_path)

# resume training for an additional 1000 epochs
num_epochs = num_epochs + 1000

train_loss, val_loss, ae_model, optimiser = train_ae(ae_model=ae_model, 
                                                      dl_train=dl_train, 
                                                      dl_val=dl_val, 
                                                      criterion=criterion, 
                                                      optimiser=optimiser, 
                                                      num_epochs=num_epochs, 
                                                      device=device,
                                                      curr_epoch=curr_epoch,
                                                      train_loss=train_loss,
                                                      val_loss=val_loss,
                                                      plot_loss=True)
```

    {"asctime": "2025-12-20T01:14:51+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder loaded from ./models/ae_model_saved.pth."}
    Training model for 4000 epochs: 100%|██████████| 1000/1000 [13:19<00:00,  1.25epoch/s, Train Loss=0.6735, Validation Loss=0.6859]
    {"asctime": "2025-12-20T01:28:10+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder training epochs 3000 to 4000 completed in 799s."}



    
![png](AE_AD_files/AE_AD_9_1.png)
    



```python
save_ae(ae_model, scaler, optimiser, num_epochs, train_loss, val_loss, save_path)
```

    {"asctime": "2025-12-20T01:28:10+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder saved in ./models/ae_model_saved.pth."}


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


    {"asctime": "2025-12-20T01:28:17+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Faultfree test RData loaded."}
    {"asctime": "2025-12-20T01:30:48+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Faulty test RData loaded."}
    {"asctime": "2025-12-20T01:31:17+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Test dataframe and dataloader returned, 20 runs for each faultNumber."}
    {"asctime": "2025-12-20T01:31:17+0800", "process": 19692, "name": "src.datapipeline", "levelname": "INFO", "message": "Test dataframe size is (403200, 55)."}


### 4.2 Process Test Data for Evaluation
Creating the `AE_eval` class object will trigger an inference run on the given test data.


```python
from src.eval_tools import AE_eval

ae_eval = AE_eval(X_test_df, dl_test, ae_model, device, scaler)
```

    Autoencoder inferencing for 53 batches: 100%|██████████| 53/53 [00:01<00:00, 30.75batch/s]
    {"asctime": "2025-12-20T01:31:20+0800", "process": 19692, "name": "src.ae_utils", "levelname": "INFO", "message": "Autoencoder inferencing for 53 test batches completed."}


Calling the `reconstruction_RMSE()` method populates a dataframe with the RMSE of all test simulation runs.


```python
ae_eval.reconstruction_RMSE()
```

    faultNumber: 100%|██████████| 21/21 [01:42<00:00,  4.90s/it]





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
      <td>0.858346</td>
      <td>1.056901</td>
      <td>0.916628</td>
      <td>0.909936</td>
      <td>1.015676</td>
      <td>1.020447</td>
      <td>0.448625</td>
      <td>0.954941</td>
      <td>...</td>
      <td>0.894276</td>
      <td>0.855289</td>
      <td>1.006675</td>
      <td>0.572446</td>
      <td>0.888005</td>
      <td>1.025035</td>
      <td>1.001644</td>
      <td>0.255462</td>
      <td>1.003629</td>
      <td>1.041911</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.800197</td>
      <td>1.070368</td>
      <td>0.858171</td>
      <td>0.856599</td>
      <td>0.996577</td>
      <td>1.018743</td>
      <td>0.394431</td>
      <td>0.924246</td>
      <td>...</td>
      <td>0.838791</td>
      <td>0.797450</td>
      <td>0.957198</td>
      <td>0.513685</td>
      <td>0.891166</td>
      <td>1.031964</td>
      <td>0.962914</td>
      <td>0.207021</td>
      <td>1.012441</td>
      <td>1.008413</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.812085</td>
      <td>1.130112</td>
      <td>0.872404</td>
      <td>0.866090</td>
      <td>0.970331</td>
      <td>0.986378</td>
      <td>0.469725</td>
      <td>0.923671</td>
      <td>...</td>
      <td>0.891232</td>
      <td>0.815816</td>
      <td>1.029090</td>
      <td>0.580082</td>
      <td>0.883356</td>
      <td>0.976178</td>
      <td>0.998885</td>
      <td>0.263494</td>
      <td>1.004581</td>
      <td>1.026165</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.865063</td>
      <td>1.131628</td>
      <td>0.892067</td>
      <td>0.876384</td>
      <td>1.008476</td>
      <td>1.022123</td>
      <td>0.479461</td>
      <td>0.915811</td>
      <td>...</td>
      <td>0.893651</td>
      <td>0.867491</td>
      <td>0.969610</td>
      <td>0.546258</td>
      <td>0.831886</td>
      <td>0.997193</td>
      <td>0.963426</td>
      <td>0.281650</td>
      <td>1.013269</td>
      <td>0.968907</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.852559</td>
      <td>1.059954</td>
      <td>0.896926</td>
      <td>0.913151</td>
      <td>0.986588</td>
      <td>1.003159</td>
      <td>0.439666</td>
      <td>0.934200</td>
      <td>...</td>
      <td>0.888783</td>
      <td>0.851015</td>
      <td>0.946380</td>
      <td>0.512763</td>
      <td>0.895703</td>
      <td>1.022570</td>
      <td>1.000608</td>
      <td>0.261032</td>
      <td>0.989208</td>
      <td>0.977163</td>
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
      <td>0.904010</td>
      <td>1.059623</td>
      <td>0.986793</td>
      <td>0.958587</td>
      <td>0.995757</td>
      <td>1.038436</td>
      <td>0.807321</td>
      <td>0.992870</td>
      <td>...</td>
      <td>0.977004</td>
      <td>0.903044</td>
      <td>0.984826</td>
      <td>4.676193</td>
      <td>0.978040</td>
      <td>0.999964</td>
      <td>1.023047</td>
      <td>0.519090</td>
      <td>1.063770</td>
      <td>1.001071</td>
    </tr>
    <tr>
      <th>416</th>
      <td>20.0</td>
      <td>17.0</td>
      <td>0.912185</td>
      <td>1.090031</td>
      <td>0.989175</td>
      <td>0.915629</td>
      <td>1.030389</td>
      <td>1.002018</td>
      <td>0.777253</td>
      <td>0.979058</td>
      <td>...</td>
      <td>0.928117</td>
      <td>0.911145</td>
      <td>0.975060</td>
      <td>4.660039</td>
      <td>0.959050</td>
      <td>0.977748</td>
      <td>1.025838</td>
      <td>0.400081</td>
      <td>1.045549</td>
      <td>1.025795</td>
    </tr>
    <tr>
      <th>417</th>
      <td>20.0</td>
      <td>18.0</td>
      <td>0.908239</td>
      <td>1.117907</td>
      <td>0.985770</td>
      <td>0.944729</td>
      <td>1.014390</td>
      <td>1.043493</td>
      <td>0.864034</td>
      <td>0.966538</td>
      <td>...</td>
      <td>0.988979</td>
      <td>0.907707</td>
      <td>0.983762</td>
      <td>4.513570</td>
      <td>0.960903</td>
      <td>0.991530</td>
      <td>1.015370</td>
      <td>0.472900</td>
      <td>1.098075</td>
      <td>1.018542</td>
    </tr>
    <tr>
      <th>418</th>
      <td>20.0</td>
      <td>19.0</td>
      <td>0.901677</td>
      <td>1.143558</td>
      <td>0.951024</td>
      <td>0.935766</td>
      <td>1.001001</td>
      <td>1.078849</td>
      <td>0.825711</td>
      <td>0.953032</td>
      <td>...</td>
      <td>0.950554</td>
      <td>0.902401</td>
      <td>0.982018</td>
      <td>4.631778</td>
      <td>0.932676</td>
      <td>1.011697</td>
      <td>1.045658</td>
      <td>0.509842</td>
      <td>1.147253</td>
      <td>1.011122</td>
    </tr>
    <tr>
      <th>419</th>
      <td>20.0</td>
      <td>20.0</td>
      <td>0.937926</td>
      <td>1.110646</td>
      <td>0.997996</td>
      <td>0.991136</td>
      <td>1.002017</td>
      <td>1.037034</td>
      <td>0.868797</td>
      <td>0.996252</td>
      <td>...</td>
      <td>0.972569</td>
      <td>0.940493</td>
      <td>1.028328</td>
      <td>4.774306</td>
      <td>1.014280</td>
      <td>0.967093</td>
      <td>0.984628</td>
      <td>0.530539</td>
      <td>1.116088</td>
      <td>1.030328</td>
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
      <td>[xmeas_1, xmeas_3, xmeas_4, xmeas_6, xmeas_7, ...</td>
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
      <td>[xmeas_11, xmeas_19, xmeas_20, xmeas_21, xmeas...</td>
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
      <td>[xmeas_1, xmeas_2, xmeas_3, xmeas_4, xmeas_7, ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>[xmeas_18, xmeas_19]</td>
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


