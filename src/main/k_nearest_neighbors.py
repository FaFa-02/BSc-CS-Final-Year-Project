import numpy as np

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
        distances = np.zeros((new_dataset.shape[0], self.X_train.shape[0]))
        y_pred_index = np.zeros((new_dataset.shape[0], self.X_train.shape[0]))
        y_pred = np.zeros(new_dataset.shape[0])

        for i in range(new_dataset.shape[0]):

            for j in range(self.X_train.shape[0]):
                # Computes distance of new and training sample
                current_dist = np.linalg.norm(new_dataset[i] - self.X_train[j])

                # Inserts distance and sample index into sorted array
                sorted_index = np.searchsorted(distances, current_dist)
                distances[i, sorted_index] = current_dist
                y_pred_index[i, np.searchsorted(distances, current_dist)] = self.y_train[j]

        # Computes mean of n closest neighbours
        for i in range(distances.shape[0]):
            y_pred[i] = np.mean(y_pred_index[0:self.n])

        print(y_pred)

        return y_pred
