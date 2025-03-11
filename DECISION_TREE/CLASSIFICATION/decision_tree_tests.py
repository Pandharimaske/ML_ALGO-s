import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys
import os

# Add the ML_ALGO-s directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DECISION_TREE.CLASSIFICATION.decision_tree_scratch import DecisionTree

def accuracy(y_true , y_pred):
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=12345)

# Train Decision Tree with Gini
tree_gini = DecisionTree(criterion="gini")
tree_gini.fit(X_train, y_train)
y_pred_gini = tree_gini.predict(X_test)
acc_gini = accuracy(y_test , y_pred_gini)

# Train Decision Tree with Entropy
tree_entropy = DecisionTree(criterion="entropy")
tree_entropy.fit(X_train, y_train)
y_pred_entropy = tree_entropy.predict(X_test)
acc_ent = accuracy(y_test , y_pred_entropy)

print("Accuracy using Gini:", acc_gini)
print("Accuracy using Entropy:", acc_ent)
