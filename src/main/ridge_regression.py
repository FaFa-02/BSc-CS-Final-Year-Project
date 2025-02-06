"""Module needed in order to compute mathematical equations with matrices"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

"""Module providing a Ridge Regression classifier with appropriate functions"""
class RidgeRegression():
    """Class representing a Ridge Regression classifier"""

    def __init__(self, penalty):
        self.X_train = None
        self.y_train = None
        self.beta_ridge_hat = None
        self.m = None
        self.n = None
        self.penalty = penalty

    def fit(self, X_train, y_train):
        """Fits the regression model to the training data."""
        # Add intercept column of 1s to feature set
        self.X_train = np.c_[np.ones((X_train.shape[0],1)), X_train]
        self.y_train = y_train

        # Stores sizes of the training set matrix
        self.m = self.X_train.shape[0]
        self.n = self.X_train.shape[1]

        # Identity matrix needed for beta_ridge_hat computation
        I = np.identity(self.n)

        self.beta_ridge_hat = ((inv((self.X_train.T).dot(self.X_train) + np.multiply(self.penalty, I) )).dot(self.X_train.T)).dot(self.y_train)
        print("beta hat values:",self.beta_ridge_hat)

    def predict(self, new_dataset):
        """Predicts values based on matrix of features from new samples."""
        # Add intercept column of 1s to feature set
        new_dataset = np.c_[np.ones((new_dataset.shape[0],1)), new_dataset]
        predictions = np.zeros(new_dataset.shape[0])

        for i in range(new_dataset.shape[0]):
            predictions[i] = new_dataset[i].dot(self.beta_ridge_hat)

        return predictions

    def tss(self, y):
        """Calculates total sum of squares from a dataset"""
        y_mean = np.mean(y)

        TSS = np.sum((y - y_mean)**2)

        return TSS

    def sse(self, y, y_pred):
        """Calculates residual sum of squares from a dataset"""
        SSE = np.sum((y - y_pred)**2)

        return SSE

    def r2(self, y, y_pred):
        """Calculates the R2 score of predicted values on true values"""
        R2 = 1 - (self.sse(y, y_pred) / self.tss(y))

        return R2

    def score(self, X_new, y_true, graph=None):
        """Predicts values and computes R Squared score for said predictions on real targets"""
        y_pred = self.predict(X_new)

        # Computes and prints R2 score
        r2_score = self.r2(y_true, y_pred)
        print("r2 =", r2_score)

        # Find largest and smallest target value, either true or predicted
        largest_label = max(np.amax(y_true), np.amax(y_pred))
        smallest_label = min(np.amin(y_true), np.amin(y_pred))

        # Creates a plot of the true vs predicted target values
        if graph is True:
            plt.scatter(y_true, y_pred, label="True", marker="*", s=30)
            plt.legend(["Predicted Values"], title=f"R2 score: {r2_score:.4f} \nAlpha: {self.penalty}", alignment='left')
            plt.axline((0,0), (1,1), color='red', label='Ideal Calibration')
            plt.xlim(smallest_label, largest_label)
            plt.ylim(smallest_label, largest_label)
            plt.xlabel("True Target")
            plt.ylabel("Predicted Target")
            plt.title("Actual vs Predicted Target")
            plt.show()

        return r2_score
