import numpy as np
from REGRESSION.regression import BaseRegression
class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X , w) + b

    def _predict(self , X , w , b):
        y_predicted = np.dot(X , w) + b
        return y_predicted
    