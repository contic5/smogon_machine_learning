# Line below disables errors
# pylint: disable=E1101
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import os

# import some data to play with
smogon = pd.read_csv('smogonmodv4.csv')
#X = smogon.ix[:,2:7].values
firstcol = 2
lastcol = 10
endcol = 11
tiers = ["Ubers", "OU", "UU", "RU", "NU", "PU"]
selectedtiernumbers = [0,5]
selectedtiers = []
for i in range(len(selectedtiernumbers)):
    selectedtiers.append(tiers[selectedtiernumbers[i]])
maxaccuracy = 0.0
maxestimationclf = ""
maxestimationclftitle=""
maxaccuratecol1name = ""
maxaccuratecol2name = ""
maxaccuratecol1 = 0
maxaccuratecol2 = 0
colnames = ["#", "Name", "Total", "Lowest Stat","HP","Highest Attack","Highest Defense","Lowest Defense","Speed"
            "Stat Standard Dev", "MinMaxDiff"]
foldername=selectedtiers[0]+"-vs-"+selectedtiers[1]
path = "smogonplots/"+foldername
if(not os.path.isdir(path)):
    original_umask = os.umask(0)
    os.makedirs(path, 0o777)

def resetvalues():
    global maxaccuracy
    global maxestimationclf
    global maxestimationclftitle
    global maxaccuratecol1
    global maxaccuratecol2
    global maxaccuratecol1name
    global maxaccuratecol2name 
    maxaccuracy = 0.0
    maxestimationclf = ""
    maxestimationclftitle=""
    maxaccuratecol1name = ""
    maxaccuratecol2name = ""
    maxaccuratecol1 = 0
    maxaccuratecol2 = 0

def getXandYdata(col1,col2,endcol):
    indices = [col1, col2]
    print("Col 1=", colnames[col1], "Col 2=", colnames[col2])
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
        for j in range(0,len(tiers)):
            if(y[i] == tiers[j]):
                y[i] = j
    y = pd.to_numeric(y)
    scaler = StandardScaler()
    scaler = scaler.fit(X, y)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X = scaler.transform(X)
    return X,y
def getclf(X,y,selectedclfval):
    svcs=[]
    C = 1.0  # SVM regularization parameter
     # SVC with linear kernel
    svcs.append(svm.SVC(kernel='linear', C=C).fit(X, y))
    # LinearSVC (linear kernel)
    svcs.append(svm.LinearSVC(C=C).fit(X, y))
    # SVC with RBF kernel
    svcs.append(svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y))
    # SVC with polynomial (degree 3) kernel
    svcs.append(svm.SVC(kernel='poly', degree=3,gamma='auto', C=C).fit(X, y))
    selectedclf=svcs[selectedclfval]
    return selectedclf


def handlemultiplot(X,y,col1,col2):
    global maxaccuracy
    global maxestimationclf
    global maxestimationclftitle
    global maxaccuratecol1
    global maxaccuratecol2
    global maxaccuratecol1name
    global maxaccuratecol2name 
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
                'SVC with polynomial(deg 3) kernel']

    maxaccvalthispair = 0
    maxaccclfthispair = ""
    maxaccclfthispairtitle = ""
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
                    label=labels, c=columns, s=5)

        #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,s=15,label=labels)
        plt.xlabel(colnames[col1])
        plt.ylabel(colnames[col2])
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
            maxaccclfthispairtitle=titles[i]
        if(score > maxaccuracy):
            maxaccuracy = score
            maxestimationclf = clf
            maxestimationclftitle=titles[i]
            maxaccuratecol1 = col1
            maxaccuratecol2 = col2
            maxaccuratecol1name = colnames[col1]
            maxaccuratecol2name = colnames[col2]
    # plt.show()
    plt.savefig(path+'/plot'+colnames[col1]+'-vs-'+colnames[col2] +
        '-'+selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
    plt.close()
    return maxaccclfthispair,maxaccclfthispairtitle
def handlesingleplot(X,y,clf,clfname,col1,col2,col1name,col2name,bestoverall=False,showplot=False):
    global maxaccuracy
    global maxestimationclf
    global maxestimationclftitle
    global maxaccuratecol1
    global maxaccuratecol2
    global maxaccuratecol1name
    global maxaccuratecol2name 

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
    poly_svc = svm.SVC(kernel='poly', degree=3,gamma='auto', C=C).fit(X, y)

    h = .01  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

    colors = ('red', 'blue', 'green', 'gray', 'cyan', 'darkviolet')
    selectedcolors = []
    selectedcolors.append(colors[selectedtiernumbers[0]])
    selectedcolors.append(colors[selectedtiernumbers[1]])
    #cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap = ListedColormap(selectedcolors)

    lightcolors = ('lightcoral', 'lightblue', 'lightgreen','lightgray', 'lightcyan', 'violet')
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
    plt.xlabel(col1name)
    plt.ylabel(col2name)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    score = clf.fit(X, y).score(X, y)
    roundscore = round(score, 2)
    plt.title(clfname+" "+str(roundscore))
    plt.legend(loc='upper left')
    print(score)
    if(score > maxaccuracy):
            maxaccuracy = score
            maxestimationclf = clf
            maxestimationclftitle=clfname
            maxaccuratecol1 = col1
            maxaccuratecol2 = col2
            maxaccuratecol1name = col1name
            maxaccuratecol2name = col2name
    if(bestoverall):
        plt.savefig(path+'/bestplot-' +selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
        plt.show()
    else:
        if(showplot):
            plt.show()
        plt.savefig(path+'/bestplot' +col1name+'-vs-'+col2name+'-'+selectedtiers[0]+"-vs-"+selectedtiers[1]+'.png')
    plt.close()
def svcallcolumns(selectedclfval=-1,specificclftitle=""):
    if(selectedclfval==-1):
        for a in range(firstcol, lastcol):
            for b in range(firstcol, a):
                X, y=getXandYdata(a,b,endcol)
                maxaccclfthispair,maxaccclfthispairtitle=handlemultiplot(X,y,a,b)
                # handle best plot for each pair
                handlesingleplot(X,y,maxaccclfthispair,maxaccclfthispairtitle,a,b,colnames[a],colnames[b])
        print("maxaccuracy=", maxaccuracy)
        print("maxestimationtype=", maxestimationclftitle)
        print("maxaccuratecol1=", maxaccuratecol1name)
        print("maxaccuratecol2=", maxaccuratecol2name)
        X,y=getXandYdata(maxaccuratecol1,maxaccuratecol2,endcol)
        handlesingleplot(X,y,maxestimationclf,maxestimationclftitle,maxaccuratecol1,maxaccuratecol2,maxaccuratecol1name,maxaccuratecol2name,bestoverall=True)
    else:
        for a in range(firstcol, lastcol):
            for b in range(firstcol, a):
                X, y=getXandYdata(a,b,endcol)
                specificclf=getclf(X,y,selectedclfval)
                # handle best plot for each pair
                handlesingleplot(X,y,specificclf,specificclftitle,a,b,colnames[a],colnames[b])
        print("maxaccuracy=", maxaccuracy)
        print("maxestimationtype=", maxestimationclftitle)
        print("maxaccuratecol1=", maxaccuratecol1name)
        print("maxaccuratecol2=", maxaccuratecol2name)
        X,y=getXandYdata(maxaccuratecol1,maxaccuratecol2,endcol)
        handlesingleplot(X,y,maxestimationclf,maxestimationclftitle,maxaccuratecol1,maxaccuratecol2,maxaccuratecol1name,maxaccuratecol2name,bestoverall=True)

def svconecolumn(selectedcol,selectedclfval=-1,specificclftitle=""):
    if(selectedclfval==-1):
        for b in range(firstcol, lastcol):
            if(b!=selectedcol):
                X, y=getXandYdata(selectedcol,b,endcol)
                maxaccclfthispair,maxaccclfthispairtitle=handlemultiplot(X,y,selectedcol,b)
                # handle best plot for each pair
                handlesingleplot(X,y,maxaccclfthispair,maxaccclfthispairtitle,selectedcol,b,colnames[selectedcol],colnames[b])
        print("maxaccuracy=", maxaccuracy)
        print("maxestimationtype=", maxestimationclftitle)
        print("maxaccuratecol1=", maxaccuratecol1name)
        print("maxaccuratecol2=", maxaccuratecol2name)
        X,y=getXandYdata(maxaccuratecol1,maxaccuratecol2,endcol)
        handlesingleplot(X,y,maxestimationclf,maxestimationclftitle,selectedcol,maxaccuratecol2,maxaccuratecol1name,maxaccuratecol2name,bestoverall=True)
    else:
        for b in range(firstcol, lastcol):
            if(b!=selectedcol):
                X, y=getXandYdata(selectedcol,b,endcol)
                specificclf=getclf(X,y,selectedclfval)
                # handle best plot for each pair
                handlesingleplot(X,y,specificclf,specificclftitle,selectedcol,b,colnames[selectedcol],colnames[b])
        print("maxaccuracy=", maxaccuracy)
        print("maxestimationtype=", maxestimationclftitle)
        print("maxaccuratecol1=", maxaccuratecol1name)
        print("maxaccuratecol2=", maxaccuratecol2name)
        X,y=getXandYdata(maxaccuratecol1,maxaccuratecol2,endcol)
        handlesingleplot(X,y,maxestimationclf,maxestimationclftitle,selectedcol,maxaccuratecol2,maxaccuratecol1name,maxaccuratecol2name,bestoverall=True)

def svcpaircolumns(col1,col2,selectedclfval=-1,specificclftitle=""):
    if(selectedclfval==-1):
        X, y=getXandYdata(col1,col2,endcol)
        maxaccclfthispair,maxaccclfthispairtitle=handlemultiplot(X,y,col1,col2)
        # handle best plot for each pair
        handlesingleplot(X,y,maxaccclfthispair,maxaccclfthispairtitle,col1,col2,colnames[col1],colnames[col2],showplot=True)
    else:
        X, y=getXandYdata(col1,col2,endcol)
        specificclf=getclf(X,y,selectedclfval)
        X, y=getXandYdata(col1,col2,endcol)
        handlesingleplot(X,y,specificclf,specificclftitle,col1,col2,colnames[col1],colnames[col2],showplot=True)

def main():
    done=0
    while(done!=1):
        ready=0
        columnoption=int(input("Enter 0 for checking all columns, 1 for checking one column"
        "\nagainst all other columns or 2 for handling a specific pair of columns: "))
        specificclfval=int(input("Enter 0 for all clfs, 1 for svc with linear kernel, 2 for linear svc,"
        "\n3 for svc with rbf kernel or 4 for svc with polynomial (degree 3) kernel: "))
        print("Checking",tiers[selectedtiernumbers[0]],"vs",tiers[selectedtiernumbers[1]])
        if(columnoption==0):
            print("Checking all combinations of columns")
            if(specificclfval==0):
                print("Checking with all clfs")
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svcallcolumns()
            else:
                titles = ['SVC with linear kernel',
                    'LinearSVC (linear kernel)',
                    'SVC with RBF kernel',
                    'SVC with polynomial(deg 3) kernel']
                title=titles[specificclfval-1]
                selectedclfval=specificclfval-1
                print("Checking with clf",title)
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svcallcolumns(selectedclfval,title)
        elif(columnoption==1):
            col1=int(input("Enter column to compare against all other columns: "))
            print("Checking",colnames[col1],"vs all other columns")
            if(specificclfval==0):
                print("Checking with all clfs")
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svconecolumn(col1)
            else:
                titles = ['SVC with linear kernel',
                    'LinearSVC (linear kernel)',
                    'SVC with RBF kernel',
                    'SVC with polynomial(deg 3) kernel']
                title=titles[specificclfval-1]
                selectedclfval=specificclfval-1
                print("Checking with clf",title)
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svconecolumn(col1,selectedclfval,title)
        else:
            col1=int(input("Enter column 1: "))
            col2=int(input("Enter column 2: "))
            print("Checking",colnames[col1],"vs",colnames[col2])
            if(specificclfval==0):
                print("Checking with all clfs")
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svcpaircolumns(col1,col2)
            else:
                titles = ['SVC with linear kernel',
                    'LinearSVC (linear kernel)',
                    'SVC with RBF kernel',
                    'SVC with polynomial(deg 3) kernel']
                title=titles[specificclfval-1]
                selectedclfval=specificclfval-1
                print("Checking with clf",title)
                ready=int(input("Type 0 to confirm or 1 to change options: "))
                if(ready==0):
                    svcpaircolumns(col1,col2,selectedclfval,title)
        resetvalues()
        done=int(input("Type 0 to continue or 1 to exit: "))
resetvalues()
main()