# Line below disables errors
# pylint: disable=E1101
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# import some data to play with
smogon = pd.read_csv('smogonmodv3nohead.csv')
#X = smogon.ix[:,2:7].values
firstcol = 2
lastcol = 7
endcol = 9
tiers = ["Uber", "OU", "UU", "RU", "NU", "PU"]
selectedtiernumbers = [1, 2]
selectedtiers = []
for i in range(len(selectedtiernumbers)):
    selectedtiers.append(tiers[selectedtiernumbers[i]])
maxaccuracy = 0.0
maxestimationtype = ""
maxaccuratecol1name = ""
maxaccuratecol2name = ""
maxaccuratecol1 = 0
maxaccuratecol2 = 0
colnames = ["#", "Name", "Total", "Lowest Stat", "Highest Attack",
            "Standard Dev", "MinMaxDiff", "Generation", "WeaknessCount", "ResistancesCount"]
for a in range(firstcol, lastcol):
    for b in range(firstcol, a):
        indices = [a, b]
        print("Col 1=", colnames[a], "Col 2=", colnames[b])
        startX = smogon.iloc[:, indices].values
        startY = smogon.iloc[:, endcol].values
        X = []
        y = []
        for i in range(0, len(startY)):
            if(startY[i] == tiers[selectedtiernumbers[0]]):
                X.append(startX[i])
                y.append(startY[i])
            elif(startY[i] == tiers[selectedtiernumbers[1]]):
                X.append(startX[i])
                y.append(startY[i])
        X[0] = pd.to_numeric(X[0])
        X[1] = pd.to_numeric(X[1])
        for i in range(0, len(y)):
            if(y[i] == "Uber"):
                y[i] = 0
            if(y[i] == "OU"):
                y[i] = 1
            if(y[i] == "UU"):
                y[i] = 2
            if(y[i] == "RU"):
                y[i] = 3
            if(y[i] == "NU"):
                y[i] = 4
            if(y[i] == "PU"):
                y[i] = 5
        y = pd.to_numeric(y)
        scaler = StandardScaler()
        scaler = scaler.fit(X, y)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        X = scaler.transform(X)
        C = 1.0  # SVM regularization parameter
        # print(X.shape)
        # print(y.shape)

        # SVC with linear kernel
        svc = svm.SVC(kernel='linear', C=C).fit(X, y)
        # LinearSVC (linear kernel)
        lin_svc = svm.LinearSVC(C=C).fit(X, y)
        # SVC with RBF kernel
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        # SVC with polynomial (degree 3) kernel
        poly_svc = svm.SVC(kernel='poly', degree=3,
                           gamma='auto', C=C).fit(X, y)

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
                  'SVC with polynomial(deg 3) kernel']

        maxaccvalthispair = 0
        maxaccclfthispair = ""
        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            colors = ('red', 'blue', 'green', 'gray', 'cyan', 'darkviolet')
            selectedcolors = []
            selectedcolors.append(colors[selectedtiernumbers[0]])
            selectedcolors.append(colors[selectedtiernumbers[1]])
            #cmap = ListedColormap(colors[:len(np.unique(y))])
            cmap = ListedColormap(selectedcolors)

            lightcolors = ('lightcoral', 'lightblue', 'lightgreen',
                           'lightgray', 'lightcyan', 'violet')
            #lightcmap = ListedColormap(lightcolors[:len(np.unique(y))])
            selectedlightcolors = []
            selectedlightcolors.append(lightcolors[selectedtiernumbers[0]])
            selectedlightcolors.append(lightcolors[selectedtiernumbers[1]])
            lightcmap = ListedColormap(selectedlightcolors)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=lightcmap, alpha=0.8)

            # Plot also the training points
            for labels, nums, columns in zip(selectedtiers, selectedtiernumbers, selectedcolors):
                plt.scatter(X[y == nums, 0],
                            X[y == nums, 1],
                            label=labels, c=columns, s=15)

            #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,s=15,label=labels)
            plt.xlabel(colnames[a])
            plt.ylabel(colnames[b])
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            # plt.xticks(())
            # plt.yticks(())
            score = clf.fit(X, y).score(X, y)
            roundscore = round(score, 2)
            plt.title(titles[i]+" "+str(roundscore))
            plt.legend(loc='upper left')
            print(score)
            if(score > maxaccvalthispair):
                maxaccvalthispair = score
                maxaccclfthispair = clf
            if(score > maxaccuracy):
                maxaccuracy = score
                maxestimationtype = titles[i]
                maxaccuratecol1 = a
                maxaccuratecol2 = b
                maxaccuratecol1name = colnames[a]
                maxaccuratecol2name = colnames[b]
        # plt.show()
        plt.savefig('smogonplots/plot'+colnames[a]+'-vs-'+colnames[b] +
                    '-'+selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
        plt.close()

        # handle best plot for each pair
        clf = maxaccclfthispair
        colors = ('red', 'blue', 'green', 'gray', 'cyan', 'darkviolet')
        selectedcolors = []
        selectedcolors.append(colors[selectedtiernumbers[0]])
        selectedcolors.append(colors[selectedtiernumbers[1]])
        #cmap = ListedColormap(colors[:len(np.unique(y))])
        cmap = ListedColormap(selectedcolors)

        lightcolors = ('lightcoral', 'lightblue', 'lightgreen',
                       'lightgray', 'lightcyan', 'violet')
        #lightcmap = ListedColormap(lightcolors[:len(np.unique(y))])
        selectedlightcolors = []
        selectedlightcolors.append(lightcolors[selectedtiernumbers[0]])
        selectedlightcolors.append(lightcolors[selectedtiernumbers[1]])
        lightcmap = ListedColormap(selectedlightcolors)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=lightcmap, alpha=0.8)

        # Plot also the training points
        for labels, nums, columns in zip(selectedtiers, selectedtiernumbers, selectedcolors):
            plt.scatter(X[y == nums, 0],
                        X[y == nums, 1],
                        label=labels, c=columns, s=10)

        #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,s=15,label=labels)
        plt.xlabel(colnames[a])
        plt.ylabel(colnames[b])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        # plt.xticks(())
        # plt.yticks(())
        score = clf.fit(X, y).score(X, y)
        roundscore = round(score, 2)
        plt.title(titles[i]+" "+str(roundscore))
        plt.legend(loc='upper left')
        print(score)
        plt.savefig('smogonplots/bestplot' +
                    colnames[a]+'-vs-'+colnames[b]+'-'+selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
print("maxaccuracy=", maxaccuracy)
print("maxestimationtype=", maxestimationtype)
print("maxaccuratecol1=", maxaccuratecol1name)
print("maxaccuratecol2=", maxaccuratecol2name)
indices = [maxaccuratecol1, maxaccuratecol2]
startX = smogon.iloc[:, indices].values
startY = smogon.iloc[:, endcol].values
X = []
y = []
for i in range(0, len(startY)):
    if(startY[i] == tiers[selectedtiernumbers[0]]):
        X.append(startX[i])
        y.append(startY[i])
    elif(startY[i] == tiers[selectedtiernumbers[1]]):
        X.append(startX[i])
        y.append(startY[i])
X[0] = pd.to_numeric(X[0])
X[1] = pd.to_numeric(X[1])
for i in range(0, len(y)):
    if(y[i] == "Uber"):
        y[i] = 0
    if(y[i] == "OU"):
        y[i] = 1
    if(y[i] == "UU"):
        y[i] = 2
    if(y[i] == "RU"):
        y[i] = 3
    if(y[i] == "NU"):
        y[i] = 4
    if(y[i] == "PU"):
        y[i] = 5
y = pd.to_numeric(y)
scaler = StandardScaler()
scaler = scaler.fit(X, y)
StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.transform(X)
C = 1.0  # SVM regularization parameter
# print(X.shape)
# print(y.shape)

# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C).fit(X, y)

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
          'SVC with polynomial(deg 3) kernel']
clf = ""
if(maxestimationtype == titles[0]):
    clf = svc
elif(maxestimationtype == titles[1]):
    clf = lin_svc
elif(maxestimationtype == titles[2]):
    clf = rbf_svc
elif(maxestimationtype == titles[3]):
    clf = poly_svc
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
# Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #plt.subplot(2, 2, i + 1)
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)

colors = ('red', 'blue', 'green', 'gray', 'cyan', 'darkviolet')
selectedcolors = []
selectedcolors.append(colors[selectedtiernumbers[0]])
selectedcolors.append(colors[selectedtiernumbers[1]])
#cmap = ListedColormap(colors[:len(np.unique(y))])
cmap = ListedColormap(selectedcolors)

lightcolors = ('lightcoral', 'lightblue', 'lightgreen',
                'lightgray', 'lightcyan', 'violet')
#lightcmap = ListedColormap(lightcolors[:len(np.unique(y))])
selectedlightcolors = []
selectedlightcolors.append(lightcolors[selectedtiernumbers[0]])
selectedlightcolors.append(lightcolors[selectedtiernumbers[1]])
lightcmap = ListedColormap(selectedlightcolors)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=lightcmap, alpha=0.8)

# Plot also the training points
for labels, nums, columns in zip(selectedtiers, selectedtiernumbers, selectedcolors):
    plt.scatter(X[y == nums, 0],X[y == nums, 1],label=labels, c=columns, s=10)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,s=15,label=labels)
plt.xlabel(colnames[maxaccuratecol1])
plt.ylabel(colnames[maxaccuratecol2])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
score = clf.fit(X, y).score(X, y)
roundscore = round(score, 2)
plt.title(titles[i]+" "+str(roundscore))
plt.legend(loc='upper left')
print(score)

plt.savefig('smogonplots/bestplot-' +selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
plt.show()


print("File saved")
