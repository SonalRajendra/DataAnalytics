import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix

# This class will perform the processings required for trainning the AI model
class DataProcessor:

    # TODO: implement later (maybe)
    #def _get_data(self):
    #    return self.data
    
    # Reading the CSV file 
    def __init__(self, file_path: str) -> None:
        self.data = pd.read_csv(file_path)

    # Getting the columns
    def get_columns(self):
        return self.data.columns.values.tolist()
    
    # Selecting the X and Y columns for further processing
    def select_X_y(self, X_columns: list, y_columns: list):
        self.X = self.data[X_columns]
        self.y = self.data[y_columns]

    # Splitting the dataset for trainning and testing
    def splitData(self, test_size: float):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=37)


class BaseMLModel:
    def fit(self, X, y):
        raise NotImplementedError("Implementation")

    def predict(self, X):
        raise NotImplementedError("Implementatiopn")
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

# Neural Network model construction, training the model and getting the prediction 
class NeuralNetworkModel(BaseMLModel):
    # Construction of Neural Network Model 
    # classifier_or_regressor parameter will define the type of data : Classification or Regression
    def __init__(self, classifier_or_regressor, activation_fn: str, no_of_layers: int, no_of_neurons: int):
        self.nn = classifier_or_regressor(
        activation=activation_fn,
        hidden_layer_sizes=(no_of_layers, no_of_neurons),
        alpha=0.0001,
        solver='lbfgs',
        max_iter=1000,
        random_state=20,
        early_stopping=True
    )
        
    # Training the Neural Network model    
    def train(self, X_train, y_train):
        self.nn.fit(X_train, y_train)

    # Predicting the result 
    def predict(self, X_test):
        self.prediction = self.nn.predict(X_test.values)


class RandomForestModel(BaseMLModel):
    # Construction of Random Forest Model 
    # classifier_or_regressor parameter will define the type of data : Classification or Regression
    def __init__(self, classifier_or_regressor, n_estimators: int, criterion: str):
        self.rf = classifier_or_regressor(
        n_estimators = n_estimators,
        criterion = criterion,
        max_depth = None,
        min_samples_split = 10,
        min_samples_leaf = 2,
        random_state = 20
    )

    # Training the Random Forest model    
    def train(self, X_train, y_train):
        self.rf.fit(X_train, y_train)

    # Predicting the result 
    def predict(self, X_test):
        self.prediction = self.rf.predict(X_test.values)

class XGBoostModel(BaseMLModel):
    def __init__(self):
        self.model = XGBoost()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

# This class will evaluate the accuracy, precision and Root Mean Square Error of the model
class Evaluation:
    # Setting the actual test value and predicted value
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    # Accuracy of the result
    def get_accuracy(self):
        return accuracy_score(self.y_test, self.y_pred)

    # Precision value
    def get_precision(self):
        return precision_score(self.y_test, self.y_pred)
    
    # Calculation of Root Mean Square Error
    def get_rmse(self):
        return np.sqrt(mean_squared_error(self.y_test, self.y_pred))

    # Confusion Matrix
    def get_confusionmatrix(self):
        return confusion_matrix(self.y_test, self.y_pred)
    
