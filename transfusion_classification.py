#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:51:18 2018

@author: akshayarka
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('/home/akshayarka/Desktop/Datasets/UCI - Blood Transfusion/transfusion.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Try Kernel SVM:
from sklearn.svm import SVC
svm_clf = SVC(C=5, kernel='rbf')
svm_clf.fit(X_train, y_train)
print(svm_clf.score(X_test, y_test))                                # 78

svm_clf2 = SVC(C=5, kernel='poly', degree=3)
svm_clf2.fit(X_train, y_train)
print(svm_clf2.score(X_test, y_test))                            # 74.6%

# Try Decision Tree:
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='entropy')
tree_clf_2 = DecisionTreeClassifier(criterion='gini')
tree_clf.fit(X_train, y_train)
tree_clf_2.fit(X_train, y_train)
print(tree_clf.score(X_test, y_test))
print(tree_clf_2.score(X_test, y_test))                         # 70% for both


# Now try Random Forest:
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_clf.fit(X_train, y_train)
print(rf_clf.score(X_test, y_test))                             # 73% here


# Go for Naive Bayes now:
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
print(nb_clf.score(X_test, y_test))                             # 75.33% for this. WINNER!!


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)