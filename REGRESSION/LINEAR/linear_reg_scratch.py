import numpy as np
import sys
import os

# Add the ML_ALGO-s directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from REGRESSION.regression import BaseRegression
class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X , w) + b

    def _predict(self , X , w , b):
        y_predicted = np.dot(X , w) + b
        return y_predicted
    