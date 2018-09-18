#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:04:55 2017

@author: giulero
"""

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
#    markers = ('o', 'x', 's', '^', 'v')
#    colors = ( 'blue', 'lightgreen', 'gray', 'cyan', 'yellow', 'green')
#    cmap = ListedColormap(colors[:len(np.unique(y))]) #unique unisce gli elementi uguali
    #plot the decison surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #crea una meshgrid con valori tra x_min e x_max con data risoluzione. arange crea un vettore tra x_min e x_max con passo resolution
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #.T->Transpose. Ravel mette in riga gli elementi
    Z = Z.reshape(xx1.shape)
    cmap_light = ListedColormap(['#005CCC','#A0CE00', '#009933', '#00FFFF', '#F0FFFF','#FF7F50','#00FF00','#B8860B','#DCDCDC','#FF69B4','#3CB371','#48D1CC'])#shape ritorna la dimensione della matrice/vettore. Reshape modella una matrice/vettore dando le dimensioni desiderate
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap_light) #contourf(x,y,z) x,y sono le coordinate della superficie        
    #plot class samples
    plt.scatter(X[:,0], X[:,1], marker='.', c = y,  edgecolor='gray', label=y)
    
def purity(pred, y, k):
    z = []
    summa = 0
    A = np.c_[pred, y]
    for j in range(k):
        z = A[A[:,0]==j, 1]
        #c = np.argmax(np.bincount(z)) #bincount restituisce il numero di volte che è ripetuto un numero. Quale numero? Quello corrispondente alla posizione all'interno del vettore.
        #con argmax ricaviamo il posto in cui c'è il numero più grande.
        #summa += len(z[z == c]) #
        summa += np.max(np.bincount(z))
    return summa/A.shape[0]
        

digits = datasets.load_digits()
X = digits.data
y = digits.target

plt.imshow(digits.images[8], interpolation="bicubic")
plt.show()

X=X[y<5]
y=y[y<5]
X = preprocessing.scale(X)
X = PCA(2).fit_transform(X)
pur_k = []
hom_k = []
norm_k = []
for k in range(3, 20):
    km = KMeans(k)
    km.fit(X)
    centroids = km.cluster_centers_
    y_p = km.predict(X)
    plot_decision_regions(X, y, km)
    plt.title('KMean -- K = %i' % k)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='black', s=500)
    if k<11: plt.savefig('KMean_k%i.pdf' % k, format='pdf') 
    plt.show()
    print('K =', k)
    pur_k.append(purity(y_p, y, k))
    hom_k.append(metrics.homogeneity_score(y ,y_p))
    norm_k.append(metrics.normalized_mutual_info_score(y, y_p))
    print("Purity = ", purity(y_p, y, k))
    print("Homogenity " + str(metrics.homogeneity_score(y ,y_p)))
    print("Normalized mutual " + str(metrics.normalized_mutual_info_score(y, y_p)))

plt.plot(range(3,20), pur_k, marker='o', label='Purity')
plt.plot(range(3,20), hom_k, marker='o', label='Homogeneity')
plt.plot(range(3,20), norm_k, marker='o', label='Normalized Mutual Information')
plt.legend(loc='best')
plt.xticks(range(3,20))
plt.savefig('images/measure_kmean.pdf', format='pdf')
plt.show()

pur_gmm = []    
hom_gmm = []
norm_gmm = []
for k in range(2, 20):
    gmm = GaussianMixture(k)
    gmm.fit(X)
    y_p = gmm.predict(X)
    plot_decision_regions(X, y, gmm)
    plt.title('GMM -- K = %i' % k)
    if k<11: plt.savefig('images/GMM_k%i.pdf' % k, format='pdf') 
    plt.show()
    print('K =', k)
    pur_gmm.append(purity(y_p, y, k))
    hom_gmm.append(metrics.homogeneity_score(y ,y_p))
    norm_gmm.append(metrics.normalized_mutual_info_score(y, y_p))
    print("Purity = ", purity(y_p, y, k))
    
plt.plot(range(2,20), pur_gmm, marker='o', label='Purity')
plt.plot(range(2,20), hom_gmm, marker='o', label='Homogeneity')
plt.plot(range(2,20), norm_gmm, marker='o', label='Normalized Mutual Information')
plt.legend(loc='best')
plt.xticks(range(3,20))
plt.savefig('images/measure_gmm.pdf', format='pdf')
plt.show() 
