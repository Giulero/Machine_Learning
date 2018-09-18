# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:43:42 2016

@author: Giulero
"""
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) #unique unisce gli elementi uguali
    #plot the decison surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #crea una meshgrid con valori tra x_min e x_max con data risoluzione. arange crea un vettore tra x_min e x_max con passo resolution
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #.T->Transpose. Ravel mette in riga gli elementi
    Z = Z.reshape(xx1.shape) #shape ritorna la dimensione della matrice/vettore. Reshape modella una matrice/vettore dando le dimensioni desiderate
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) #contourf(x,y,z) x,y sono le coordinate della superficie        
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c ='', alpha=1.0, linewidths=1, marker='o', label='test set')
