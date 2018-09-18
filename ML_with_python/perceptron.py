# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 18:09:54 2016

@author: Giulero
"""

import numpy as np

class Perceptron(object):
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1]) #X.shape fornisce le dimensioni della matrice X, [0] indica le righe, [1] le colonne. Np.zeros(n) crea un array lungo n zeri
        self.errors_ = [] #le variabili segnate con _ richiamano gli altri metodi dell'oggetto
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y): #zip accoppia l'n-esima riga della matrice X con l'n-esimo elemento del vettore y
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
                      
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1) #where returns value [1, -1] depends on condition
        