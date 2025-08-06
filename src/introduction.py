from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

iris = load_iris()
X = iris['data']
y = iris['target']

print("\nCampos to dataset Iris ({0}):\n{1}\n"
          .format("iris.keys()", iris.keys()))

print("\nParte da descrição do dataset Iris ({0}):\n{1}\n"
          .format("iris['DESCR'][:471]", iris['DESCR'][:491]))

print("\nNomes das classes ({0}):\n{1}\n"
          .format("iris['target_names']", iris['target_names']))

print("\nNomes das features ({0}):\n{1}\n"
          .format("iris['feature_names']", iris['feature_names']))


X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

print("\nBase de treinamento ({0}):\n{1}\n".format("X_train.shape", X_train.shape))

print("\nBase de teste ({0}):\n{1}\n".format("X_test.shape", X_test.shape))

iris_dataframe = pd.DataFrame(np.c_[iris['data'], iris['target']],
					columns=np.append(iris['feature_names'], 'target'))

ax2 = pd.plotting.scatter_matrix(iris_dataframe.iloc[:,:4], figsize=(11, 11), c=y, marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.figure()
ax3 = pd.plotting.parallel_coordinates(iris_dataframe, "target")


