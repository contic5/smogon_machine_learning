import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(6, 5))

df = pd.read_csv(filepath_or_buffer='smogonmodv2nohead.csv',header=None, sep=',')

X = df.iloc[0:,2:6].values
y = df.iloc[0:,8].values

'''
startX = df.iloc[0:,2:6].values
startY = df.iloc[0:,6].values
X=[]
y=[]
for i in range(0,len(startY)):
    if(startY[i]=="Uber"):
        X.append(startX[i])
        y.append(startY[i])
    elif(startY[i]=="PU"):
        X.append(startX[i])
        y.append(startY[i])
print("len y=",len(y))
'''
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
'''

X_scaled = StandardScaler().fit_transform(X)
svd = TruncatedSVD(n_components=2)
Y_fitted = svd.fit_transform(X_scaled)

for labels, columns in zip(('Uber','OU','UU','RU','NU','PU'),('red','orange','yellow','green','blue','purple')):
    plt.scatter(
    Y_fitted[y==labels, 0],
    Y_fitted[y==labels, 1],
    label=labels,c=columns, s=15)
    
plt.xlabel('Principal Component 1(ATK,SPA,SPD)')
plt.ylabel('Principal Component 2(HP,DEF,SPDF)')
plt.legend(loc='best')
plt.title("SVD On Iris Data", fontsize=20)

print(svd)
plt.show()