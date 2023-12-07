"""Module needed in order to compute mathematical equations with matrices"""
import numpy as np
from numpy.linalg import inv

"""Module providing a linear regression classifier with appropriate functions"""
class LinearRegressionClassifier():
    """Class representing a linear regression classifier"""

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

        beta_ridge_hat = ((inv((self.X_train.T).dot(self.X_train) + self.penalty * I)).dot(self.X_train.T)).dot(y_train)
        print(beta_ridge_hat)
