import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_reg_scratch import LogisticRegression

bc = datasets.load_breast_cancer()
X , y = bc.data , bc.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

def accuracy(y_true , y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regressor = LogisticRegression()
regressor.fit(X_train , y_train)
predictions = regressor.predict(X_test)

print("LR Classification Accuracy" , accuracy(y_test , predictions))
