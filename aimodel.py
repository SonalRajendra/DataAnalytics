import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# This class will perform the processings required for trainning the AI model
class DataProcessor:
    # TODO: implement later (maybe)
    # def _get_data(self):
    #    return self.data

    def __init__(self, file_path: str) -> None:
        """
        Initialize DataProcessor object with a file path. It is reading the data from csv file.

        Parameters:
        - file_path (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(file_path)

    def get_columns(self):
        """
        It returns the column names of the dataset.

        Returns:
        - list: List of column names.
        """
        return self.data.columns.values.tolist()

    def select_X_y(self, X_columns: list, y_columns: list):
        """
        Select X and y columns from the given dataset.

        Parameters:
        - X_columns (list): List of column names to be selected as features (X).
        - y_columns (list): List of column names to be selected as target (y).
        """
        self.X = self.data[X_columns]
        self.y = self.data[y_columns]
        self.y = self.y.values.ravel()

    def splitData(self, test_size: float):
        """
        Split the dataset into training and testing datasets.

        Parameters:
        - test_size (float): The proportion of the dataset to include in the test split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=37
        )

    def scaleData(self):
        """
        Scales the dataset to unit variance by removing the mean. It is required if the data is not pre-processed.
        """
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


class BaseMLModel:
    def fit(self, X, y):
        raise NotImplementedError("Implementation")

    def predict(self, X):
        raise NotImplementedError("Implementation")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


class NeuralNetworkModel(BaseMLModel):
    """
    A wrapper class for Neural Network model, supporting both classification and regression.
    """

    def __init__(
        self,
        classifier_or_regressor,
        activation_fn: str,
        no_of_layers: int,
        no_of_neurons: int,
    ):
        """
        Initialize the NeuralNetwork object with the specified parameters.

        Parameters:
        - classifier_or_regressor: It will define the type of data : Classification or Regression.
        The underlying Neural Network model to be used, either MLPClassifier or MLPRegressor.
        - activation_fn (str): The activation function for the hidden layers. Supported activation functions include 'logistic', 'tanh', and 'relu'.
        - no_of_layers (int): The number of hidden layers in the Neural Network.
        - no_of_neurons (int): The number of neurons in each hidden layer.
        - alpha (float, optional, default=0.0001): Regularization parameter. It penalizes large weights in the model, helping to prevent overfitting.
        - solver (str, optional, default='lbfgs'): The optimization algorithm used to train the Neural Network.
        Supported solvers include 'lbfgs', 'adam', and 'sgd'.
        - max_iter (int, optional, default=100000): The maximum number of iterations for training the Neural Network. It controls the maximum number of iterations the optimization algorithm will run.
        - random_state (int, optional, default=20): The seed used by the random number generator. It ensures reproducibility of results when training the model.
        - early_stopping (bool, optional, default=True): Whether to use early stopping to terminate training when validation score stops improving. Early stopping helps prevent overfitting by stopping training when the model starts to perform worse on validation data.
        """
        self.nn = classifier_or_regressor(
            activation=activation_fn,
            hidden_layer_sizes=(no_of_layers, no_of_neurons),
            alpha=0.0001,
            solver="lbfgs",
            max_iter=100000,
            random_state=20,
            early_stopping=True,
        )

    # Training the Neural Network model
    def train(self, X_train, y_train):
        """
        Trains the Neural Network model according to the given training data.

        Parameters:
        - X_train: The selected features of the training data.
        - y_train: The target value of the training data.
        """
        self.nn.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target for the given test data.

        Parameters:
        - X_test: The selected features of the test data.

        Returns:
        - prediction: The predicted target of the test data.
        """
        self.prediction = self.nn.predict(X_test)


class RandomForestModel(BaseMLModel):
    """
    A wrapper class for Random Forest model, supporting both classification and regression.
    """

    def __init__(self, classifier_or_regressor, n_estimators: int, criterion: str):
        """
        Initialize the RandomForest object with the specified parameters.

        Parameters:
        - classifier_or_regressor: It will define the type of data : Classification or Regression.
        The underlying Random Forest model to be used, either RandomForestClassifier or RandomForestRegressor.
        - n_estimators (int): The number of trees in the forest.
        - criterion (str): The function to measure the quality of a split.
        Supported criteria are "gini", "entropy", "log_loss" for Classification.
        Supported criteria are "friedman_mse", "squared_error", "poisson" for Regression.
        - min_samples_split (int): The minimum number of samples required to split an internal node
        - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        - max_depth: Maximum depth of the individual regression estimators.
        """
        self.rf = classifier_or_regressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=20,
        )

    def train(self, X_train, y_train):
        """
        Train the Random Forest model according to the given training data.

        Parameters:
        - X_train: The selected features of the training data.
        - y_train: The target value of the training data.
        """
        self.rf.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target for the given test data.

        Parameters:
        - X_test: The selected features of the test data.

        Returns:
        - prediction: The predicted target of the test data.
        """
        self.prediction = self.rf.predict(X_test)


class XGBoostModel(BaseMLModel):
    """
    A wrapper class for XGBoost model, supporting both classification and regression.
    """

    # TODO: to be implemented
    def __init__(self):
        self.xgboost = XGBoost()  # noqa: F821

    def train(self, X_train, y_train):
        """
        Train the XGBoost model according to the given training data.

        Parameters:
        - X_train: The selected features of the training data.
        - y_train: The target value of the training data.
        """
        self.xgboost.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target for the given test data.

        Parameters:
        - X_test: The selected features of the test data.

        Returns:
        - prediction: The predicted target of the test data.
        """
        self.prediction = self.xgboost.predict(X_test)


# This class will evaluate the accuracy, precision and Root Mean Square Error of the model
class Evaluation:
    """
    This class evaluates the performance of the above developed machine learning models.
    """

    def __init__(self, y_test, y_pred):
        """
        Initialize Evaluation object with target (y_test) and predicted (y_pred) values.

        Parameters:
        - y_test (array-like): True target values.
        - y_pred (array-like): Predicted values.
        """
        self.y_test = y_test
        self.y_pred = y_pred

    def get_accuracy(self):
        """
        Calculate the accuracy of the model.

        Returns:
        - float: Accuracy score.
        """
        return accuracy_score(self.y_test, self.y_pred)

    def get_precision(self):
        """
        Calculate the precision value of the model.

        Returns:
        - float: Precision score.
        """
        return precision_score(self.y_test, self.y_pred)

    def get_rmse(self):
        """
        Calculation of Root Mean Square Error (RMSE) of the model.

        Returns:
        - float: RMSE value.
        """
        return np.sqrt(mean_squared_error(self.y_test, self.y_pred))

    def get_confusionmatrix(self):
        """
        Calculate the confusion matrix of the model.

        Returns:
        - array: Confusion matrix.
        """
        return confusion_matrix(self.y_test, self.y_pred)

    def plot_confusion_matrix(self, conf_matrix):
        """
        Plot the confusion matrix of the model.

        Plot: Confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)
