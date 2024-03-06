import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from aimodel import DataProcessor, Evaluation, NeuralNetworkModel, RandomForestModel


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a CSV file for testing
        self.file_path = "test_data.csv"
        data = {
            "testfeature1": [1, 2, 3, 4, 5],
            "testfeature2": [6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.file_path, index=False)

    def tearDown(self):
        # Delete the CSV file after testing
        import os

        os.remove(self.file_path)

    def test_init(self):
        # Test the initoalizationof DataProcessor object with file path
        data_processor = DataProcessor(self.file_path)
        self.assertIsInstance(data_processor.data, pd.DataFrame)

    def test_get_columns(self):
        # Test if get_columns method returns column names
        data_processor = DataProcessor(self.file_path)
        columns = data_processor.get_columns()
        self.assertListEqual(columns, ["testfeature1", "testfeature2", "target"])

    def test_select_X_y(self):
        # Test if select_X_y method selects X and y columns correctly
        data_processor = DataProcessor(self.file_path)
        data_processor.select_X_y(["testfeature1", "testfeature2"], ["target"])
        self.assertListEqual(
            data_processor.X.columns.tolist(), ["testfeature1", "testfeature2"]
        )
        self.assertListEqual(data_processor.y.tolist(), [0, 1, 0, 1, 0])

    def test_splitData(self):
        # Test if splitData method splits data into train and test sets
        data_processor = DataProcessor(self.file_path)
        data_processor.select_X_y(["testfeature1", "testfeature2"], ["target"])
        data_processor.splitData(test_size=0.4)
        self.assertEqual(len(data_processor.X_train), 3)
        self.assertEqual(len(data_processor.X_test), 2)
        self.assertEqual(len(data_processor.y_train), 3)
        self.assertEqual(len(data_processor.y_test), 2)

    def test_scaleData(self):
        # Test if scaleData method scales the data
        data_processor = DataProcessor(self.file_path)
        data_processor.select_X_y(["testfeature1", "testfeature2"], ["target"])
        data_processor.splitData(test_size=0.4)
        data_processor.scaleData()
        self.assertTrue(
            (data_processor.X_train >= 0).all() and (data_processor.X_train <= 1).all()
        )
        self.assertFalse(
            (data_processor.X_test >= 0).all() and (data_processor.X_test <= 1).all()
        )


class TestNeuralNetworkModel(unittest.TestCase):
    def test_classifier_creation(self):
        classifier = MLPClassifier
        model = NeuralNetworkModel(
            classifier, activation_fn="relu", no_of_layers=2, no_of_neurons=100
        )
        self.assertIsInstance(model.nn, MLPClassifier)

    def test_regressor_creation(self):
        regressor = MLPRegressor
        model = NeuralNetworkModel(
            regressor, activation_fn="relu", no_of_layers=2, no_of_neurons=100
        )
        self.assertIsInstance(model.nn, MLPRegressor)

    def test_training(self):
        classifier = MLPClassifier
        model = NeuralNetworkModel(
            classifier, activation_fn="relu", no_of_layers=2, no_of_neurons=100
        )
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        model.train(X_train, y_train)
        self.assertIsNotNone(model.nn.coefs_)

    def test_prediction(self):
        classifier = MLPClassifier
        model = NeuralNetworkModel(
            classifier, activation_fn="relu", no_of_layers=2, no_of_neurons=100
        )
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        X_test = [[-1, -1], [2, 2], [3, 2]]
        model.train(X_train, y_train)
        model.predict(X_test)
        self.assertIsNotNone(model.prediction)

    def test_regressor_training(self):
        regressor = MLPRegressor
        model = NeuralNetworkModel(
            regressor, activation_fn="tanh", no_of_layers=2, no_of_neurons=100
        )
        X_train = [[0, 0], [1, 1]]
        y_train = [0, 1]
        model.train(X_train, y_train)
        self.assertIsNotNone(model.nn.coefs_)

    def test_regressor_prediction(self):
        regressor = MLPRegressor
        model = NeuralNetworkModel(
            regressor, activation_fn="tanh", no_of_layers=2, no_of_neurons=100
        )
        X_train = [[0, 0], [1, 1]]
        y_train = [0, 1]
        X_test = [[2.0, 2.0], [-1.0, -2.0]]
        model.train(X_train, y_train)
        model.predict(X_test)
        self.assertIsNotNone(model.prediction)


class TestRandomForestModel(unittest.TestCase):
    def test_classifier_creation_rf(self):
        classifier = RandomForestClassifier
        model = RandomForestModel(classifier, n_estimators=100, criterion="gini")
        self.assertIsInstance(model.rf, RandomForestClassifier)

    def test_regressor_creation_rf(self):
        regressor = RandomForestRegressor
        model = RandomForestModel(regressor, n_estimators=100, criterion="mse")
        self.assertIsInstance(model.rf, RandomForestRegressor)

    def test_classifier_training_rf(self):
        classifier = RandomForestClassifier
        model = RandomForestModel(classifier, n_estimators=100, criterion="gini")
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        model.train(X_train, y_train)
        self.assertIsNotNone(model.rf.estimators_)

    def test_classifier_prediction_rf(self):
        classifier = RandomForestClassifier
        model = RandomForestModel(classifier, n_estimators=100, criterion="gini")
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        X_test = [[-1, -1], [2, 2], [3, 2]]
        model.train(X_train, y_train)
        model.predict(X_test)
        self.assertIsNotNone(model.prediction)

    def test_regressor_training_rf(self):
        regressor = RandomForestRegressor
        model = RandomForestModel(regressor, n_estimators=100, criterion="friedman_mse")
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        model.train(X_train, y_train)
        self.assertIsNotNone(model.rf.estimators_)

    def test_regressor_prediction_rf(self):
        regressor = RandomForestRegressor
        model = RandomForestModel(regressor, n_estimators=100, criterion="friedman_mse")
        X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y_train = [-1, -1, -1, 1, 1, 1]
        X_test = [[-1, -1], [2, 2], [3, 2]]
        model.train(X_train, y_train)
        model.predict(X_test)
        self.assertIsNotNone(model.prediction)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Setting up test data
        self.y_test = np.array([0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 0, 1, 0])

    def test_accuracy(self):
        evaluation = Evaluation(self.y_test, self.y_pred)
        accuracy = evaluation.get_accuracy()
        # 4 out of 5 predictions are correct
        self.assertEqual(accuracy, 0.8)

    def test_precision(self):
        evaluation = Evaluation(self.y_test, self.y_pred)
        precision = evaluation.get_precision()
        # Precision is 1.0 because there are no false positives
        self.assertAlmostEqual(precision, 1.0)

    def test_rmse(self):
        evaluation = Evaluation(self.y_test, self.y_pred)
        rmse = evaluation.get_rmse()
        # Root mean squared error for the last sample
        self.assertAlmostEqual(rmse, np.sqrt(0.2))

    def test_confusion_matrix(self):
        evaluation = Evaluation(self.y_test, self.y_pred)
        confusion_matrix_actual = evaluation.get_confusionmatrix()
        # Confusion matrix for the test data
        confusion_matrix_expected = np.array([[2, 0], [1, 2]])
        np.testing.assert_array_equal(
            confusion_matrix_actual, confusion_matrix_expected
        )


if __name__ == "__main__":
    unittest.main()
