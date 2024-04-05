"""Module providing a K Nearest Neighbour Regression model with appropriate functions"""
class KNearestNeighbors():
    """Class representing a KNN regression model"""

    def __init__(self, n = 3):
        self.n = n

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train