import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os

# Add the ML_ALGO-s directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from DECISION_TREE.REGRESSION.decision_tree_scratch import DecisionTreeRegressor  # Import the regression version

# Load the Diabetes dataset
data = datasets.load_diabetes()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# Train Decision Tree Regressor using MSE
tree_mse = DecisionTreeRegressor(criterion="mse")  # Can also use "variance"
tree_mse.fit(X_train, y_train)
y_pred_mse = tree_mse.predict(X_test)
mse_mse = mean_squared_error(y_test, y_pred_mse)

# Train Decision Tree Regressor using Variance
tree_var = DecisionTreeRegressor(criterion="variance")
tree_var.fit(X_train, y_train)
y_pred_var = tree_var.predict(X_test)
mse_var = mean_squared_error(y_test, y_pred_var)

print("MSE using Mean Squared Error:", mse_mse)
print("MSE using Variance:", mse_var)