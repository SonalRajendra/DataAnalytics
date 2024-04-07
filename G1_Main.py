import numpy as np                                                  # Importing numpy library for numerical operations
import pandas as pd                                                 # Importing pandas library for data manipulation
from scipy.stats import zscore                                      # Importing zscore function from scipy.stats
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter                              # Importing specific functions
from sklearn.ensemble import IsolationForest 


"""
Classes and Methods Used:
- Class: OutlierDetection
  - Method: detect_outliers_zscore_drop
          Parameter: Threshold
  - Method: detect_outliers_iqr_drop
          Parameter: Threshold
  - Method: detect_outliers_isof_drop
          Parameter: Contamination Rate

- Class: Plotter
  - Method: plot_interpolated

- Class: Smoothing
  - Method: moving_average
          Parameter: Window Length
  - Method: savitzky_golay
          Parameter: Window Length
          Parameter: Order

- Class: Interpolation
  - Method: interpolation
          Parameter: Method
"""
 
"""
Class for outlier detection methods
"""
class OutlierDetection:
    def __init__(self, df):                                         # Initializing class with a DataFrame
        self.df = df                                                # Assigning the DataFrame to an attribute
    """
    Method for detecting outliers using Z-score
    """ 
    def detect_outliers_zscore_drop(self, column_names, threshold):
        cleaned_data_df = self.df.copy()                            # Make a copy of the DataFrame
        for column in column_names:                                 # Iterate through specified column names
            column_data = self.df[column]                           # Extract data from the specified column
            z_scores = zscore(column_data)                          # Calculate z-scores for the column data
            z_score_outliers = np.abs(z_scores) > threshold         # Identify outliers based on z-score and threshold                                                           
            cleaned_data_df.loc[z_score_outliers, column] = np.nan  # Replace outliers with NaN values
            
        return cleaned_data_df                                      # Return the DataFrame with outliers replaced by NaN values
    """
    Method for detecting outliers using Inter Quartile Range 
    """
    def detect_outliers_iqr_drop(self, column_names, threshold):
        cleaned_data_df = self.df.copy()                            # Make a copy of the DataFrame
        for column in column_names:                                 # Iterate through specified column names
            column_data = self.df[column]                           # Extract data from the specified column
            q1 = column_data.quantile(0.25)                         # Calculating first quartile
            q3 = column_data.quantile(0.75)                         # Calculating third quartile
            iqr = q3 - q1                                           # Calculating interquartile range
                                                                    # Identify outliers based on IQR and threshold
            iqr_outliers = self.df[(column_data < q1 - threshold * iqr) | (column_data > q3 + threshold * iqr)].index.tolist()
                                                                    # Replace outliers with NaN values
            cleaned_data_df.loc[iqr_outliers, column] = np.nan
        return cleaned_data_df

    """
    Method for detecting outliers using Isolation Forest
    """

    def detect_outliers_isof_drop(self, column_names, contamination_rate):
        cleaned_data_df = self.df.copy()                             # Make a copy of the DataFrame
        for column_name in column_names:                             # Iterate through specified column names
            column_data = self.df[column_name]                       # Extracting column data
            isolation_forest = IsolationForest(max_samples=100, contamination=contamination_rate, random_state=42)
            isolation_forest_outliers = isolation_forest.fit_predict(column_data.values.reshape(-1, 1))
            outliers_mask = isolation_forest_outliers == -1          # Ceates a boolean mask where True indicates the presence of an outlier                                     
            cleaned_data_df.loc[outliers_mask, column_name] = np.nan # Replace outliers with NaN values
        return cleaned_data_df

"""
Class for Plotter
"""

class Plotter:
    def __init__(self, df):
        self.df = pd.DataFrame(df)                                  # Initializing class with a DataFrame
                                                                    
    def plot_interpolated(self, df, selected_column):               # Function to plot interpolated data
                                                                    
        fig1, ax1 = plt.subplots()                                  # Creating a new figure and axes
                                                                    # Plotting interpolated data
        ax1.plot(df.index, df[selected_column], label='Interpolated', color='blue')  
                                                                    # Scatter plot of original data points
        ax1.scatter(df.index, df[selected_column], color='red', label='Original')  
        ax1.set_xlabel('Index')                                     # Setting x-axis label
        ax1.set_ylabel(selected_column)                             # Setting y-axis label
        ax1.set_title('Interpolated Values')                        # Setting plot title
        ax1.legend()                                                # Adding legend
        return fig1                                                 # Returning the figure


"""
Class for data smoothing methods
"""

class Smoothing:
    
    def __init__(self, df):
        self.df = pd.DataFrame(df)                                  # Initializing class with a DataFrame

    """
    Method for applying moving average filter to smooth data
    """
    def moving_average(self, column_name, filter_length):
                                                                    # Output DataFrame to store results
        df_var = pd.DataFrame()
                                                                    # Iterate over each column in the DataFrame
        df_var = self.df.copy()                                     # Creating a copy of the DataFrame
        for column in column_name:                                  # Iterating through specified column names
                                                                    # Calculating moving average and updating the column with smoothed values
            df_var[column] = self.df[column].rolling(filter_length, min_periods=1).mean()
        return df_var                                               # Returning the smoothed DataFrame

    """
    Method for applying Savitzky-Golay filter to smooth data
    """
    def savitzky_golay(self, column_name, filter_length, order):
        df_var = self.df.copy()                                     # Creating a copy of the DataFrame
        for column in column_name:                                  # Iterating through specified column names
                                                                    # Applying Savitzky-Golay filter and updating the column with smoothed values
            df_var[column] = savgol_filter(df_var[column], filter_length, order)
        return df_var                                               # Returning the smoothed DataFrame

"""
Class for interpolation methods
"""

class Interpolation:
    def __init__(self, available_methods=['linear', 'quadratic', 'cubic']):
        self.available_methods = available_methods                 # Initializing available interpolation methods

    def interpolation(self, df, column_names, method='linear'):
        for column_name in column_names:                           # Iterate through specified column names
                                                                   # Performing interpolation on the specified column
            if method in self.available_methods:                   # Checking if the interpolation method is valid
                                                                   # Interpolating using specified method
                df[column_name] = df[[column_name]].interpolate(method=method)
                                                                   # Filling any remaining NaN values
                df[column_name] = df[[column_name]].interpolate(method='bfill')
                df[column_name] = df[[column_name]].interpolate(methpd='ffill')  
            else:
                                                                   # Raise an error if the interpolation method is invalid
                raise ValueError("Invalid interpolation method. Please choose one of the following: {}".format(self.available_methods))
        return df                                                  # Returning the interpolated DataFrame

