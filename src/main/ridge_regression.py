"""Module needed in order to compute mathematical equations with matrices"""
import numpy as np
from numpy.linalg import inv

"""Module providing a Ridge Regression classifier with appropriate functions"""
class RidgeRegressionClassifier():
    """Class representing a Ridge Regression classifier"""

    def __init__(self, penalty):
        self.X_train = None
        self.y_train = None
        self.m = None
        self.n = None
        self.penalty = penalty

    def fit(self, X_train, y_train):
        """Fits the regression model to the training data."""
        self.X_train = X_train
        self.y_train = y_train

        # Stores sizes of the training set matrix
        self.m = X_train.shape[0]
        self.n = X_train.shape[1]

        # Identity matrix needed for beta_ridge_hat computation
        I = np.identity(self.n)

        self.beta_ridge_hat = ((inv((self.X_train.T).dot(self.X_train) + self.penalty * I)).dot(self.X_train.T)).dot(y_train)
        print(self.beta_ridge_hat)

    def predict(self, new_dataset):
        """Predicts values based on matrix of features from new samples."""
        print(type(new_dataset))
        predictions = np.zeros(new_dataset.shape[0])

        for i in range(new_dataset.shape[0]):
            predictions[i] = new_dataset[i].dot(self.beta_ridge_hat)
        print(predictions)
        return predictions

    def rss(self, X_new, y_true):
        """Predicts values and computes RSS score for said predictions on real targets"""
        y_pred = self.predict(X_new)

        rss_score = np.sum((y_true - y_pred)** 2)

        return rss_score
    
    def tss(self, X_new, y_true):
        y_pred = self.predict(X_new)

        y_pred_avg = np.average(y_pred)

        tss_score = np.sum((y_true - y_pred_avg)** 2)

        return tss_score

