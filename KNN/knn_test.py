import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000' , '#00FF00' , '#0000FF'])

iris = datasets.load_iris()
X , y = iris.data , iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# print(f"shape of X_train :{X_train.shape}")
# print(f"first row of X_train :{X_train[0]}")

# print(f"shape of y_train :{y_train.shape}")
# print(f"first row of y_train :{y_train[0]}")

# print(f"Scatter Plot of X_train")
# plt.figure()
# plt.scatter(X[: , 2] , X[: , 3] , c = y , cmap=cmap , edgecolors='k' , s = 20)
# plt.show()

print("Started Model Training")
from knn_scratch import KNN
clf = KNN(k = 3)
clf.fit(X_train , y_train)
print("Started Model Testing")
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(f"Acc of Model : {acc}")

