import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
titanic_data = pd.read_csv(r'C:\Users\thoma\Downloads\titanic_excel.csv')
df = pd.DataFrame(titanic_data)
dfx = df.drop(columns=['Name','Sex','Survived'])
sy=df.Survived

from sklearn.model_selection import train_test_split
dfx_train, dfx_test, sy_train, sy_test= train_test_split(dfx,sy)
dfx_train, dfx_test, sy_train, sy_test;

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = le.fit(sy_train)
le.classes_;
y_train = le.transform(sy_train)
sy_train, y_train;
n1 = preprocessing.MinMaxScaler()
n1 = n1.fit(dfx_train.to_numpy())
X_train = n1.transform(dfx_train.to_numpy())

from sklearn.neighbors import KNeighborsClassifier
n = 81
score_test = []
score_train = []
k_vals = []
#looping through knn values to create the axis for the graphs
for i in range(1,n):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train,y_train)
    X_test = n1.transform(dfx_test.to_numpy())
    y_test = le.transform(sy_test.to_numpy())
    k_vals.append(i)
    score_test.append(knn.score(X_test,y_test))
    score_train.append(knn.score(X_train,y_train))
#graph plotting
xplot = np.array(k_vals)
yplot = np.array(score_test)
plt.title("scores for K values 1-80, Test dataset")
plt.ylabel('score')
plt.xlabel('K')
plt.plot(xplot, yplot)
plt.show()

xplot = np.array(k_vals)
yplot2 = np.array(score_train)
plt.title("scores for K values 1-80, Train dataset")
plt.ylabel('score')
plt.xlabel('K')
plt.plot(xplot, yplot2)
plt.show()
#confusion matrix implimentation
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.svm import SVC
new = []
y_true = []
y_pred= knn.predict(X_test)
i = 0
while i < 222:
    b = y_pred[i]
    a = y_test[i]
    y_true.append(a)
    new.append(b)
    i +=1
matrix = confusion_matrix(y_true,new)
tn, fp, fn, tp = matrix.ravel()




