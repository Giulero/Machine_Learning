# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:07:42 2016

@author: the_s
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from perceptron import Perceptron
from decisionboundary import plot_decision_regions
from adeline import AdalineGD

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y=df.iloc[0:100, 4].values #values() returns a list of all the values available in a given dictionary.
y=np.where( y == 'Iris-setosa', -1, 1)
X=df.iloc[0:100, [0,2]].values
plt.figure(1)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal lenght')
plt.ylabel('petal lenght')
plt.legend(loc='upper left')
clf=Perceptron(eta=0.01, n_iter=10)
clf.fit(X, y)

plt.figure(2)
plot_decision_regions(X, y, clf)
plt.xlabel('sepal lenght[cm]')
plt.ylabel('petal lenght[cm]')
plt.legend(loc='upper left')
#standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

clf2 = AdalineGD(n_iter = 15, eta = 0.01).fit(X_std, y)
plt.figure(3)
plot_decision_regions(X_std, y, clf2)

#Cost graph

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].plot(range(1, len(clf2.cost_) + 1), np.log10(clf2.cost_), marker = 'o')
ax[0].set_xlabel('Ephocs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01')
clf3 = clf2 = AdalineGD(n_iter = 15, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(clf3.cost_) + 1), np.log10(clf3.cost_), marker = 'o')
ax[1].set_xlabel('Ephocs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - learning rate 0.0001')
plt.show()

