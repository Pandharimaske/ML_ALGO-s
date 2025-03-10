import numpy as np
import sys
import os

# Add the ML_ALGO-s directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from REGRESSION.regression import BaseRegression
class LogisticRegression(BaseRegression):

    def _approximation(self, X, w, b):
        linear_model = np.dot(X , w) + b
        return self._sigmoid(linear_model)
        
    def _predict(self, X , w , b):
        linear_model = np.dot(X , w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    