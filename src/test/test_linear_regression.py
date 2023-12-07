import sys
import unittest
import numpy as np

sys.path.append('..')

from src.main.linear_regression import LinearRegressionClassifier


class TestLinRegClassFit(unittest.TestCase):

    def setUp(self):
        self.lin_reg = LinearRegressionClassifier(1)
        self.X_train = np.array([[0.11425,0.00,13.890,1,0.5500,6.3730,92.40,3.3633,5,276.0,16.40,393.74,10.50], [ 6.96215,0.00,18.100,0,0.7000,5.7130,97.00,1.9265,24,666.0,20.20,394.43,17.11], [12.80230,0.00,18.100,0,0.7400,5.8540,96.60,1.8956,24,666.0,20.20,240.52,23.79]])
        self.y_train = np.array([23.00, 15.10, 10.80])
        self.lin_reg.fit(self.X_train, self.y_train)
        self.X_test = np.array([[0.11425,0.00,13.890,1,0.5500,6.3730,92.40,3.3633,5,276.0,16.40,393.74,10.50]])
        

    def test_lin_reg_fit(self):
        self.assertIsNotNone(self.lin_reg.X_train)
        self.assertIsNotNone(self.lin_reg.y_train)

    def test_lin_reg_predict(self):
        prediction = self.lin_reg.predict(self.X_test)
        self.assertIsNotNone(prediction)


if __name__ == "__main__":
    unittest.main()
