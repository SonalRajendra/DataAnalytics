import pandas as pd  # Importing pandas library for data manipulation
from sklearn.ensemble import IsolationForest
import numpy as np  # Importing numpy library for numerical operations
from scipy.signal import savgol_filter  # Importing specific functions 
import matplotlib.pyplot as plt
from scipy.stats import zscore


# Class for outlier detection methods
class OutlierDetection:
    def __init__(self, df):
        self.df = df  # Initializing class with a DataFrame

    # Method for detecting outliers using z-scores
    def detect_outliers_zscore(self, column_name, threshold):
        column_data = self.df[column_name]  # Extracting column data
        z_scores = zscore(column_data)
        z_score_outliers = (np.abs(z_scores) > threshold)
    #     mean_value = column_data.mean()  # Calculating mean of the column
    #     std_dev = column_data.std()  # Calculating standard deviation of the column
    #     z_scores = (column_data - mean_value) / std_dev  # Calculating z-scores
        z_score_outliers = self.df[abs(z_scores) > threshold].index.tolist()  # Finding indices of outliers based on threshold
        return z_score_outliers  # Returning indices of outliers


    # Function to remove outliers from the data
    def remove_outliers(self, column_name, outliers):
        column_data = self.df[column_name]  # Extracting column data
        data_cleaned = column_data.copy()
        data_cleaned[outliers] = np.nan
        #data_cleaned = data_cleaned.dropna()
        return data_cleaned

    # Method for detecting outliers using interquartile range (IQR)
    def detect_outliers_iqr(self, column_name, threshold):
        column_data = self.df[column_name]  # Extracting column data
        q1 = column_data.quantile(0.25)  # Calculating first quartile
        q3 = column_data.quantile(0.75)  # Calculating third quartile
        iqr = q3 - q1  # Calculating interquartile range
        # Finding indices of outliers based on IQR and threshold
        iqr_outliers = self.df[(column_data < q1 - threshold * iqr) | (column_data > q3 + threshold * iqr)].index.tolist()
        return iqr_outliers  # Returning indices of outliers
    
    def detect_outliers_isof(self, column_name, contamination_rate):
        column_data = self.df[column_name]  # Extracting column data
        isolation_forest = IsolationForest(max_samples=100, contamination=contamination_rate, random_state=42)  # Adjust contamination based on expected outlier rate
        isolation_forest_outliers = isolation_forest.fit_predict(column_data.values.reshape(-1, 1))
        return isolation_forest_outliers

class Plotter:
    def __init__(self, df):
        self.df = df 

    def plot_isof(self, df, selected_column, outlier_result):
        # Plot the isolation forest data
        fig, ax = plt.subplots()
        ax.scatter(df.index[outlier_result == -1], df[selected_column][outlier_result == -1], color='red', label='Errors')
        ax.scatter(df.index[outlier_result == 1], df[selected_column][outlier_result == 1], color='green', label='Result Data')
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_column)
        ax.set_title('Isolation Forest Values')
        ax.legend()
        return fig
    
    # Function to plot original data with outliers highlighted
    def plot_data_with_outliers(self, column_name, outliers):
        column_data = self.df[column_name]  # Extracting column data
        fig0, ax0 = plt.subplots()
        ax0.plot(column_data.index, column_data, color='blue', label='Original Data')
        ax0.plot(column_data.index[outliers], column_data[outliers], 'ro', label='Outliers')
        ax0.set_xlabel("Index")
        ax0.set_ylabel(column_name)
        ax0.set_title('Original Data with Outliers Highlighted')
        ax0.legend()
        return fig0
    
    def plot_interpolated(self, df, selected_column):
        # Plot the interpolated data
        fig1, ax1 = plt.subplots()
        ax1.plot(df.index, df[selected_column], label='Interpolated', color='blue')
        ax1.scatter(df.index, df[selected_column], color='red', label='Original')
        ax1.set_xlabel('Index')
        ax1.set_ylabel(selected_column)
        ax1.set_title('Interpolated Values')
        ax1.legend()
        return fig1

# Class for data smoothing methods
class Smoothing:
    def __init__(self, df):
        self.df = df

    # Method for applying moving average filter to smooth data
    def moving_average(self, column_name, filter_length):
        df_var = self.df.copy()  # Creating a copy of the DataFrame
        # Calculating moving average and updating the column with smoothed values
        df_var[column_name] = self.df[column_name].rolling(filter_length).mean()
        df_var.dropna(inplace=True)  # Dropping NaN values
        df_var.reset_index(drop=True, inplace=True)  # Resetting index
        return df_var  # Returning the smoothed DataFrame

    # Method for applying Savitzky-Golay filter to smooth data
    def savitzky_golay(self, column_name, filter_length, order):
        df_var = self.df.copy()  # Creating a copy of the DataFrame
        # Applying Savitzky-Golay filter and updating the column with smoothed values
        df_var[column_name] = savgol_filter(df_var[column_name], filter_length, order)
        df_var.dropna(inplace=True)  # Dropping NaN values
        df_var.reset_index(drop=True, inplace=True)  # Resetting index
        return df_var  # Returning the smoothed DataFrame

# Class for interpolation methods
class Interpolation:
    def __init__(self, available_methods=['linear', 'quadratic', 'cubic']):
        self.available_methods = available_methods

    def interpolation(self, df, column_name, method='linear'):
        # Performing interpolation on the specified column
        if method in self.available_methods:
            df[column_name] = df[[column_name]].interpolate(method=method)
            df[column_name] = df[[column_name]].interpolate(method='bfill')
        else:
            raise ValueError("Invalid interpolation method. Please choose one of the following: {}".format(self.available_methods))
        return df  # Returning the interpolated DataFrame
    
# Class for Date/Time Converter
class TimeConverter():
   
    def Time_Converter(df):
       
        # Iterates through each column in the DataFrame
        for column_name in df.columns:
            # Checks if the column data type is string
            if df[column_name].dtype == 'object':
                try:
                    # Converts compatible columns to datetime format 
                    df[column_name] = pd.to_datetime(df[column_name]) #This line of code takes the values in the selected column  
                                                                      #of the DataFrame df, converts them to datetime format
                except:
                    pass
            else:
                # Leaves other columns unchanged
                df[column_name] = df[column_name]
        return df

