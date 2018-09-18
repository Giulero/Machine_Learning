# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 17:03:34 2016

@author: Giulero"""

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from decisionboundary import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Accuracy: ', accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std)) #Mette in ordine verticalmente le matrici
y_combined = np.hstack((y_train, y_test))   #Mette in ordine in orizzontale
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal lenght[std]')
plt.ylabel('petal width[std]')
plt.legend(loc='upper left')
plt.show()
lr = LogisticRegression(C=1000.0, random_state = 0)
lr.fit(X_train_std, y_train)
y1_pred = lr.predict(X_test_std)
print('Accuracy: ', accuracy_score(y_test, y1_pred))
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal lenght [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal lenght [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal lenght [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal lenght [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()
