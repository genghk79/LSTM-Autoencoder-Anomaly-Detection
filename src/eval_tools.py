import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from src.datapipeline import tensor_to_df
from src.ae_utils import AEUtils


class AE_eval():
    """Evaluation class for autoencoder-based anomaly detection.

    Performs inference, calculates reconstruction RMSE, and provides visualization
    tools for analyzing autoencoder performance on test data.

    Args:
        X_test_df: Test DataFrame containing original data.
        dl_test: DataLoader for test data.
        ae_model: Trained LSTM autoencoder model.
        scaler: StandardScaler used for data normalization.

    Attributes:
        X_test_df: Original test DataFrame.
        scaler: StandardScaler for normalization.
        ae_model: Trained autoencoder model.
        device: Computation device.
        dl_test: Test DataLoader.
        recon_tensor: Reconstructed data as tensor.
        recon_df: Reconstructed data as DataFrame.
        RMSE: DataFrame containing RMSE values per feature and simulation run.
    """

    def __init__(
            self, 
            X_test_df: pd.DataFrame, 
            dl_test: DataLoader,
            ae_model: AEUtils, 
            scaler: StandardScaler
        ) -> None:

        self.X_test_df = X_test_df
        self.scaler = scaler
        self.ae_model = ae_model
        self.dl_test = dl_test
        
        self.recon_tensor = self.ae_model.inference(dl_test)
        self.recon_df = tensor_to_df(X_test_df, self.recon_tensor, self.scaler)

        self.RMSE = pd.DataFrame([])

    def reconstruction_RMSE(self) -> pd.DataFrame:
        """Calculate RMSE between original and reconstructed data.

        Computes root mean squared error for each feature across all test
        simulation runs and fault numbers. Results are stored in self.RMSE.

        Returns:
            DataFrame with columns: faultNumber, simulationRun, and RMSE for each feature.
        """
        
        # standardise test data (so that MSE will be same scale for all features)
        temp = self.X_test_df.iloc[:, 3:].values
        temp = self.scaler.transform(temp)
        X_test_scaled = self.X_test_df.copy()
        X_test_scaled.iloc[:, 3:] = temp

        # standardise reconstructed data
        temp = self.recon_df.iloc[:, 3:].values
        temp = self.scaler.transform(temp)
        recon_scaled = self.recon_df.copy()
        recon_scaled.iloc[:, 3:] = temp

        # prepare to calculate MSE for each test simulationRun, for each faultNumber
        num_runs = X_test_scaled['simulationRun'].max() - X_test_scaled['simulationRun'].min() + 1
        num_faults = X_test_scaled['faultNumber'].max() - X_test_scaled['faultNumber'].min() + 1
        RMSEs = np.zeros((num_runs * num_faults, len(X_test_scaled.columns)-1))

        # loop through all faultNumber and simulationRun
        pbar = tqdm(range(X_test_scaled['faultNumber'].min(), 
                          X_test_scaled['faultNumber'].max()+1), 
                          desc="faultNumber")
        for fault_n in pbar:
            for run_n in range(X_test_scaled['simulationRun'].min(), X_test_scaled['simulationRun'].max()+1):
                a = fault_n - X_test_scaled['faultNumber'].min()
                b = run_n - X_test_scaled['simulationRun'].min()
                RMSEs[b + a*num_runs, 0] = fault_n
                RMSEs[b + a*num_runs, 1] = run_n
                # loop through each feature
                for col_n in range(3,len(X_test_scaled.columns)):
                    sq_err = (X_test_scaled.loc[(X_test_scaled['simulationRun']==run_n) & (X_test_scaled['faultNumber']==fault_n), 
                                                X_test_scaled.columns[col_n]] - 
                            recon_scaled.loc[(recon_scaled['simulationRun']==run_n) & (recon_scaled['faultNumber']==fault_n), 
                                            recon_scaled.columns[col_n]]) ** 2
                    
                    RMSEs[b + a*num_runs, col_n-1] = np.sqrt(np.mean(sq_err))

        self.RMSE = pd.DataFrame(RMSEs, columns=['faultNumber', 'simulationRun']+self.X_test_df.columns[3:].tolist())
        return self.RMSE

    def plot_RMSE_distributions(self, faultNumber: int) -> None:
        """Plot RMSE distributions for each feature for a given fault number.

        Creates a box plot showing the distribution of RMSE values across all
        simulation runs for each feature.

        Args:
            faultNumber: The fault number to plot (0 for fault-free).

        Raises:
            ValueError: If faultNumber does not exist in the RMSE DataFrame.
        """

        if self.RMSE.empty:
            print("Call reconstruction_RSME() first to calculate RMSE values.")
            return None
        
        A = self.RMSE.drop(['simulationRun'], axis=1)
        if A.loc[A['faultNumber']==faultNumber].empty:
            raise ValueError(f"faultNumber = {faultNumber} does not exist in provided MSE dataframe")

        plt.figure(figsize=[12,4])
        ax = sns.boxplot(A.loc[A['faultNumber']==faultNumber].drop('faultNumber', axis=1)) 
        tick_locations = ax.get_xticks()
        tick_labels = ax.get_xticklabels()
        ax.set_xticks(tick_locations)
        _ = ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_ylabel('RMSE')
        ax.set_title(f"RMSE distribution for faultNumber {faultNumber}")
        plt.show()

    def find_high_RMSE(self, RMSE_threshold: float) -> pd.DataFrame:
        """Identify features with RMSE above threshold for each fault number.

        Args:
            RMSE_threshold: RMSE value above which features are considered anomalous.

        Returns:
            DataFrame with columns: faultNumber and highMSEcolumns (list of feature names).
        """

        # get feature names
        col_names = self.RMSE.columns[2:]
        over_threshold = []

        for fault_n in range(int(self.RMSE['faultNumber'].min()), int(self.RMSE['faultNumber'].max())+1):
            # list to store feature that have RMSE above threshold for this faultNumber
            cols = []
            # loop through all features and append feature name to list if RMSE above threshold 
            for col in col_names:
                if self.RMSE.loc[self.RMSE['faultNumber']==fault_n, col].mean() >= RMSE_threshold:
                    cols.append(col)
            over_threshold.append({'faultNumber': fault_n, 'highMSEcolumns': cols})

        return pd.DataFrame(over_threshold)

    def plot_feature(self, feature: Union[int, str], faultNumber: int, simulationRun: Optional[int] = None) -> None:
        """Plot original vs reconstructed data for a specific feature.

        Visualizes how well the autoencoder reconstructs a particular feature
        for a given fault number, either averaged across all runs or for a specific run.

        Args:
            feature: Feature name (string) or column index (int) to plot.
            faultNumber: The fault number to plot (0 for fault-free).
            simulationRun: Optional specific simulation run number. If None, plots min-max across all runs.
        """

        # allows feature to be called by name or column number
        if isinstance(feature, int):
            feature = self.X_test_df.columns[feature]

        A = self.X_test_df.loc[self.X_test_df['faultNumber']==faultNumber]
        B = self.recon_df.loc[self.recon_df['faultNumber']==faultNumber]

        # if no simulationRun number given, do min-max plot of the features across all runs
        if simulationRun:
            sns.lineplot(data=A.loc[A['simulationRun']==simulationRun], 
                        x='sample', y=A[feature], label='original data')
            sns.lineplot(data=B.loc[B['simulationRun']==simulationRun], 
                        x='sample', y=B[feature], label='reconstructed data')
            plt.title(f"Feature plot for simulationRun {simulationRun} of faultNumber {faultNumber}")
        else:
            sns.lineplot(data=A, 
                        x='sample', y=A[feature], label='original data')
            sns.lineplot(data=B, 
                        x='sample', y=B[feature], label='reconstructed data')
            plt.title(f"Feature plot of faultNumber {faultNumber}")
            
        plt.legend()
        plt.show()