import sys
import unittest

sys.path.append('..')

from src.main.linear_regression import LinearRegressionClassifier


class TestLinRegClassFit(unittest.TestCase):

    def setUp(self):
        self.lin_reg = LinearRegressionClassifier()

    def test_linReg_fit(self):
        self.assertIsNotNone(self.lin_reg.X_train)

if __name__ == "__main__":
    unittest.main()
