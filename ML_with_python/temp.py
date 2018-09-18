# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 17:46:03 2016

@author: the_s
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from decisionboundary import plot_decision_regions

np.random.seed(0)
X_xor = np.random.rand(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0.5, X_xor[:, 1] > 0.5)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.legend()
plt.show()
svm = SVC(kernel='rbf', random_state=0, gamma=0.4, C=50.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
