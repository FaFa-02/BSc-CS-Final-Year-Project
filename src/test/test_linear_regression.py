import sys
import unittest
import numpy as np

sys.path.append('..')

from src.main.linear_regression import LinearRegressionClassifier


class TestLinRegClassFit(unittest.TestCase):

    def setUp(self):
        self.lin_reg = LinearRegressionClassifier()
        X_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        y_train = np.array([1, 2])
        test_dataset = np.array([[1, 2, 3, 4]])
        self.lin_reg.fit(X_train, y_train)

    def test_lin_reg_fit(self):
        self.assertIsNotNone(self.lin_reg.X_train)
        self.assertIsNotNone(self.lin_reg.y_train)

    def test_lin_reg_predict(self):
        self.assertIsNotNone(self.lin_reg.predict(self.test_dataset))


if __name__ == "__main__":
    unittest.main()
