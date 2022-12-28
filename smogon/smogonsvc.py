#Line below disables errors
# pylint: disable=E1101
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd

df = pd.read_csv('smogonmodnoheadthreetiers.csv', header=None)
y = df.iloc[:, 7].values
X = df.iloc[:, [2, 4]].values


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
minval=10**300
print("Calculating svc")
svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)

# create a mesh to plot in
print("creating mesh")
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

print("creating subplot")
plt.subplot(1, 1, 1)
print("Making predictions")
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
print("reshaping z")
Z = Z.reshape(xx.shape)
print("finishing predictions")
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Total Stats')
plt.ylabel('Lowest Stat')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
print("Showing graph")
plt.show()