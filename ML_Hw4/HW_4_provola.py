# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:35:56 2017

@author: the_s
"""


import numpy as np
import pylab as pl

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data[:, :2]  # Take only 2 dimensions
y = iris.target
X = X[y > 0]
y = y[y > 0]
y -= 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))

np.random.seed(0)

gamma_range = [1e-1, 1, 1e1]
C_range = [1, 1e2, 1e4]

pl.figure()
k = 1

for C in C_range:
    for gamma in gamma_range:
        # fit the model
        clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
        clf.fit(X, y)

        # plot the decision function for each datapoint on the grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

#        pl.subplot(3, 3, k)
        pl.title("gamma %.1f, C %.2f" % (gamma, C))
        k += 1
        pl.pcolormesh(xx, yy, -Z, cmap=pl.cm.jet)
        pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.jet)
        pl.xticks(())
        pl.yticks(())
        pl.axis('tight')
        pl.show()

pl.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
pl.show()