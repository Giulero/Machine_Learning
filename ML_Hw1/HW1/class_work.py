# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 17:03:34 2016

@author: Giulero"""

from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
#import seaborn as sns

class HW1:

    def __init__(self, img_folder, n_img):
      print("---- Path loaded -----")
      for i in n_img:
          print("   ----- Image n.", i, "-----")
      self.img_folder = img_folder+'obj'
      self.n_img = n_img
      self.X = []
      self.X_std = []
      self.y = []
      self.img_list = []

    def X_(self):
        img_list = []
        img_data_raveled = []
        for num in self.n_img:
            self.img_list += glob.glob(self.img_folder+str(num)+'_*')
        for filename in self.img_list:
            im = np.asarray(Image.open(filename).convert("RGB"))
            im_raveled = np.ravel(im)
            img_data_raveled.append(im_raveled)
        #for filename in self.img_list:
        #    img_data_raveled.append(np.ravel(np.asarray(Image.open(filename),'r').convert("RGB")))
        self.X = np.array(img_data_raveled).reshape((len(self.img_list)), -1)
        self.X_std = preprocessing.scale(self.X)
        return self

    def Y(self):
        for num in range(0,len(self.n_img)):
            self.y += [num]*int(len(self.img_list)/len(self.n_img))
        return self

    def PCA(self, value):
        self.value_ = value
        print("\n----- "+str(value-1)+"-"+str(value)+" principal component -----")
        pca = PCA(value)
        self.X_PCA1 = pca.fit_transform(self.X_std)
        self.X_PCA = self.X_PCA1[:,value-2:value]
        return self

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_PCA, self.y, test_size=0.5)
        return self

    def fit(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_PCA, self.y, test_size=0.5, random_state=55)
        self.clf = GaussianNB()
        self.clf.fit(X_train, y_train)
        return self

    def covariance_plot(self):
#        cm = np.corrcoef(self.X_PCA1)
#        sns.set(font_scale=1.5)
#        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15})
#        sns.set(style='whitegrid', context='notebook')
        cov_mat = np.cov(self.X_PCA1.T)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
#        cm = np.corrcoef(eig_vec)
#        hm = sns.heatmap(cm)
        cum_exp = np.cumsum(eig_val/np.sum(eig_val))
#        plt.plot(range(1, len(eig_val)+1), eig_val/np.sum(eig_val), marker='o')
        plt.bar(range(1, len(eig_val)+1), eig_val/np.sum(eig_val), align='center', label='Individual explained variance')
        plt.step(range(1, len(eig_val)+1), cum_exp, where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='upper left')
        plt.savefig('components.pdf', format='pdf', dpi=2000)
        plt.show()
#        sns.reset_orig()
    
    def accuracy(self):
        self.pred = self.clf.predict(self.X_test)
        print ("----- Accuracy:", accuracy_score(self.y_test, self.pred),"-----")
        return  accuracy_score(self.y_test, self.pred)

    def scatter_plot(self):
        markers = ('o', 'x', 's', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])
        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X_PCA[self.y == cl, 0], y=self.X_PCA[self.y == cl, 1], alpha=0.8, c=cmap(idx), label='Obj '+str(self.n_img[cl]))
        plt.legend(loc='upper left')
        #plt.show()

    def plt(self):
        markers = ('o', 'x', 's', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])
        x1_min, x1_max = self.X_PCA[:,0].min() - 1, self.X_PCA[:,0].max()+1
        x2_min, x2_max = self.X_PCA[:,1].min() - 1, self.X_PCA[:,1].max()+1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
        #crea una meshgrid con valori tra x_min e x_max con data risoluzione. arange crea un vettore tra x_min e x_max con passo resolution
        Z = self.clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #.T->Transpose. Ravel mette in riga gli elementi
        Z = Z.reshape(xx1.shape) #shape ritorna la dimensione della matrice/vettore. Reshape modella una matrice/vettore dando le dimensioni desiderate
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        self.scatter_plot()
        #plt.title('Imgs classification with GaussianNB clf')
        plt.title('Accuracy : '+ str(self.accuracy()))
        plt.savefig('PCA'+str(self.value_)+'.pdf', format='pdf', dpi = 2000)
        #plt.legend(loc='upper left')
        plt.show()
