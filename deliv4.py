# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

# Read Titanic data from a CSV file into a Pandas DataFrame
titanic_data = pd.read_csv(r'titanic_excel.csv')
df = pd.DataFrame(titanic_data)

# Drop specific columns from the DataFrame
dfx = df.drop(columns=['Name', 'Sex', 'Survived'])
sy = df.Survived

# Import train_test_split function from scikit-learn to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
dfx_train, dfx_test, sy_train, sy_test = train_test_split(dfx, sy)

# Import preprocessing functions for label encoding and feature scaling
from sklearn import preprocessing

# Label encode the 'Survived' column for training data
le = preprocessing.LabelEncoder()
le = le.fit(sy_train)

# Store the encoded classes
le.classes_

# Transform the target labels
y_train = le.transform(sy_train)

# Feature scaling using Min-Max scaling
n1 = preprocessing.MinMaxScaler()
n1 = n1.fit(dfx_train.to_numpy())

# Transform the training data
X_train = n1.transform(dfx_train.to_numpy())

# Import the KNeighborsClassifier for k-nearest neighbors classification
from sklearn.neighbors import KNeighborsClassifier

# Initialize variables for k values, test and training scores
n = 81
score_test = []
score_train = []
k_vals = []

# Loop through different k values to build KNN models and evaluate them
for i in range(1, n):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train, y_train)

    # Transform the test data and labels
    X_test = n1.transform(dfx_test.to_numpy())
    y_test = le.transform(sy_test.to_numpy())

    k_vals.append(i)
    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))

# Create a plot for the test scores with different k values
xplot = np.array(k_vals)
yplot = np.array(score_test)
plt.title("Scores for K values 1-80, Test dataset")
plt.ylabel('Score')
plt.xlabel('K')
plt.plot(xplot, yplot)
plt.show()

# Create a plot for the training scores with different k values
xplot = np.array(k_vals)
yplot2 = np.array(score_train)
plt.title("Scores for K values 1-80, Train dataset")
plt.ylabel('Score')
plt.xlabel('K')
plt.plot(xplot, yplot2)
plt.show()

# Import confusion matrix related functions
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Initialize lists to store true and predicted values
new = []
y_true = []

# Predict on the test data using the best k (from the previous loop)
y_pred = knn.predict(X_test)
i = 0
while i < 222:
    b = y_pred[i]
    a = y_test[i]
    y_true.append(a)
    new.append(b)
    i += 1

# Calculate the confusion matrix
matrix = confusion_matrix(y_true, new)
tn, fp, fn, tp = matrix.ravel()


