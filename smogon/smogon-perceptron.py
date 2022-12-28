import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v','x')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','violet')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot class samples
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
      alpha=0.8, c=cmap(idx),
      marker=markers[idx], label=cl)

#main
indices=[2,4]
df = pd.read_csv('smogonmodv2nohead.csv', header=None)
y = df.iloc[0:, 6].values
'''
y = np.where(y == 'Uber')
y = np.where(y == 'OU')
y = np.where(y == 'UU')
y = np.where(y == 'RU')
y = np.where(y == 'NU')
y = np.where(y == 'PU')
'''
for i in range(0,len(y)):
   if(y[i]=="Uber"):
      y[i]=0
   if(y[i]=="OU"):
      y[i]=1
   if(y[i]=="UU"):
      y[i]=2
   if(y[i]=="RU"):
      y[i]=3
   if(y[i]=="NU"):
      y[i]=4
   if(y[i]=="PU"):
      y[i]=5
y=pd.to_numeric(y)
X = df.iloc[0:, indices].values
X[0]=pd.to_numeric(X[0])
X[1]=pd.to_numeric(X[1])
scaler = StandardScaler()
scaler=scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
X=scaler.transform(X)

#Scatterplot
plt.scatter(X[0:42, 0], X[0:42, 1], color='red', marker='o', label='Uber')
plt.scatter(X[42:115, 0], X[42:115, 1], color='orange', marker='o', label='OU')
plt.scatter(X[115:201, 0], X[115:201, 1], color='yellow', marker='o', label='UU')
plt.scatter(X[201:256, 0], X[201:256, 1], color='green', marker='o', label='RU')
plt.scatter(X[256:321, 0], X[256:321, 1], color='blue', marker='o', label='NU')
plt.scatter(X[321:, 0], X[321:, 1], color='purple', marker='o', label='PU')
plt.xlabel('Base Stat Total')
plt.ylabel('Highest Attack')
plt.legend(loc='upper left')
plt.show()

pn = Perceptron(0.8, 10)
pn.fit(X, y)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

# import Perceptron from perceptron.py
pn = Perceptron(0.8, 10)
pn.fit(X, y)
plot_decision_regions(X, y, classifier=pn)
plt.xlabel('Highest Attack')
plt.ylabel('Highest Defense')
plt.legend(loc='upper left')
plt.show()