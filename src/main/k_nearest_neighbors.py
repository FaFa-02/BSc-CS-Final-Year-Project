import numpy as np
import matplotlib.pyplot as plt

"""Module providing a K Nearest Neighbour Regression model with appropriate functions"""
class KNearestNeighbors():
    """Class representing a KNN regression model"""

    def __init__(self, n = 3):
        self.n = n

    def fit(self, X_train, y_train):
        """Fits the regression model to the training data."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, new_dataset):
        """Predicts new samples using KNN regression algorithm"""
        y_pred = np.array([])

        for i in range(new_dataset.shape[0]):
            distance = np.array([])
            y_pred_index = np.array([])
            for j in range(self.X_train.shape[0]):
                # Computes distance of new and training sample
                current_dist = np.linalg.norm(new_dataset[i] - self.X_train[j])

                # Inserts distance and sample index into sorted array
                sorted_index = np.searchsorted(distance, current_dist)
                distance =  np.insert(distance, sorted_index, current_dist, axis=None)
                y_pred_index = np.insert(y_pred_index, sorted_index, self.y_train[j], axis=None)

            # Compute average of n neighbours and adds to prediction output
            y_pred = np.append(y_pred, np.mean(y_pred_index[0:self.n]), axis=None)

        print(y_pred)

        return y_pred

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
            plt.legend(["Predicted Values"], title=f"R2 score: {r2_score:.4f} \nn neighbours: {self.n}", alignment='left')
            plt.axline((0,0), (1,1), color='red', label='Ideal Calibration')
            plt.xlim(smallest_label, largest_label)
            plt.ylim(smallest_label, largest_label)
            plt.xlabel("True Target")
            plt.ylabel("Predicted Target")
            plt.title("Actual vs Predicted Target")
            plt.show()

        return r2_score        
