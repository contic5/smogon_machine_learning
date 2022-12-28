#Line below disables errors
# pylint: disable=E1101
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# import some data to play with
smogon = pd.read_csv('smogonmodv2nohead3tiers.csv')
#X = smogon.ix[:,2:7].values
indices=[2,3]
labels=["OU","NU"]
X = smogon.iloc[:,indices].values
X[0]=pd.to_numeric(X[0])
X[1]=pd.to_numeric(X[1])
y = smogon.iloc[:,6].values
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
scaler = StandardScaler()
scaler=scaler.fit(X,y)
StandardScaler(copy=True, with_mean=True, with_std=True)
X=scaler.transform(X)
C = 1.0  # SVM regularization parameter
print(X.shape)
print(y.shape)

# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3,gamma='auto', C=C).fit(X, y)

h = .01  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	            np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC with linear kernel',
	  'LinearSVC (linear kernel)',
	  'SVC with RBF kernel',
	  'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    colors = ('red', 'blue', 'green', 'gray', 'cyan','violet')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    lightcolors = ('lightcoral', 'lightblue', 'lightgreen', 'gray', 'cyan','violet')
    lightcmap = ListedColormap(lightcolors[:len(np.unique(y))])

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=lightcmap, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,label=labels)
    plt.xlabel('Base Stat Total')
    plt.ylabel('Lowest Stat')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xticks(())
    #plt.yticks(())
    plt.title(titles[i])
    print(clf.fit(X, y).score(X, y))
plt.show()