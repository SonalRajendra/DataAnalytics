import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod


class DataProcessor:
    """
    This class performs the processings required for trainning the AI model
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize DataProcessor object with a pandas dataframe

        Parameters:
        - df (pd.Dataframe): Dataframe containing the dataset.
        """
        self.data = df

    def get_columns(self):
        """
        It returns the column names of the dataset.

        Returns:
        - list: List of column names.
        """
        return self.data.columns.values.tolist()
    
    def select_X_y(
        self, X_columns: list, y_columns: list
    ):
        """
        Select X and y columns from the given dataset.

        Parameters:
        - X_columns (list): List of column names to be selected as features (X).
        - y_columns (list): List of column names to be selected as target (y).
        """
        self.X = self.data[X_columns]
        self.y = self.data[y_columns]

    def splitData(self, test_size: float, classifier_or_regressor: str):
        """
        Split the dataset into training and testing datasets.

        Parameters:
        - test_size (float): The proportion of the dataset to include in the test split.
        """
        if classifier_or_regressor == "Classification":
            self.y = self.y.values.ravel()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=37
            )

    def scaleData(self):
        """
        Scales the dataset to unit variance by removing the mean. It is required if the data is not pre-processed.
        """
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


class BaseMLModel(ABC):
    #@abstractmethod
    #def fit(self, X, y):
    #    """
    #    Train the model.
#
    #    Parameters:
    #    X (array-like): Training data.
    #    y (array-like): Target values.
#
    #    Returns:
    #    None
    #    """
    #    pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the trained model.

        Parameters:
        X (array-like): Data to predict on.

        Returns:
        array-like: Predicted values.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model using the provided training data.

        Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Target values.

        Returns:
        None
        """
        pass


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
            alpha=0.001,
            solver="lbfgs",
            max_iter=100000,
            random_state=42,
            early_stopping=True,
        )

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
        - max_depth: Maximum depth of the individual estimators.
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

    def __init__(
        self,
        classifier_or_regressor,
        n_estimators: int,
        learning_rate: int,
        objective: str,
    ):
        """
        Initialize the XGBoost object with the specified parameters.

        Parameters:
        - classifier_or_regressor: It will define the type of data : Classification or Regression.
        The underlying XGBoost model to be used, either XGBClassifier or XGBRegressor.
        - n_estimators (int): The number of trees in the forest.
        - objective (str): Specify the learning task and the corresponding learning objective.
        Supported objective are "binary:logistic", "binary:logitraw", "binary:hinge" for Classification.
        Supported objective are "reg:squarederror", "reg:squaredlogerror", "reg:logistic" for Regression.
        - max_depth: Maximum depth of the individual estimators.
        """
        self.xgb = classifier_or_regressor(
            objective=objective,
            learning_rate=learning_rate,
            max_depth=4,
            n_estimators=n_estimators,
            random_state=20,
        )

    def train(self, X_train, y_train):
        """
        Train the XGBoost model according to the given training data.

        Parameters:
        - X_train: The selected features of the training data.
        - y_train: The target value of the training data.
        """
        self.xgb.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target for the given test data.

        Parameters:
        - X_test: The selected features of the test data.

        Returns:
        - prediction: The predicted target of the test data.
        """
        self.prediction = self.xgb.predict(X_test)


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
        return np.sqrt(
            mean_squared_error(self.y_test, self.y_pred, multioutput="uniform_average")
        )

    def get_recall_score(self):
        """
        Calculation of Recall Score of the model.

        Returns:
        - float: recall score
        """
        return recall_score(y_true=self.y_test, y_pred=self.y_pred)

    def get_mae(self):
        """
        Calculation of Mean Absolute Error (MAE) of the model.

        Returns:
        - float: MAE value.
        """
        return mean_absolute_error(
            self.y_test, self.y_pred, multioutput="uniform_average"
        )

    def get_r2(self):
        """
        Calculation of R-squared (R²) Score of the model.

        Returns:
        - float: R² value.
        """
        return r2_score(self.y_test, self.y_pred)

    def get_confusionmatrix(self):
        """
        Calculate the confusion matrix of the model.

        Returns:
        - array: Confusion matrix.
        """
        return confusion_matrix(self.y_test, self.y_pred)

    def scatter_plot_predicted_vs_actual(self):
        """
        Scattered plot of Predicted and Actual values 
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        # Scatter plot of actual values
        ax.scatter(self.y_test, self.y_test, color='black', label='Actual Values')
        # Scatter plot of predicted values
        ax.scatter(self.y_test, self.y_pred, color='red', alpha=0.5, label='Predicted Values')
        ax.set_title("Scatter Plot of Predicted vs. Actual Values")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True)
        return fig

