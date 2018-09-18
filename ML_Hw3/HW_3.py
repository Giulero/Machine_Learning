# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:09:08 2017

@author: Giulero
"""

from sklearn import neighbors, datasets
#from sklearn.neighbors import NearestNeighbors
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from decisionboundary import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print('----Load data----')
iris = datasets.load_iris()
X = iris.data
y = iris.target

sc = StandardScaler()
sc.fit(X)
pca=PCA(2)
pca.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
clf = neighbors.KNeighborsClassifier(3)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
pred = clf.predict(X_test)
##plot_decision_regions(X, y, classifier=clf)
##Plot the decision boundary 
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1    
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.figure()
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
#plt.pcolortmesh(xx, yy, Z, cmap=cmap_light)
x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
#crea una meshgrid con valori tra x_min e x_max con data risoluzione. arange crea un vettore tra x_min e x_max con passo resolution
Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #.T->Transpose. Ravel mette in riga gli elementi
Z = Z.reshape(xx1.shape) #shape ritorna la dimensione della matrice/vettore. Reshape modella una matrice/vettore dando le dimensioni desiderate
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap_light)