# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 17:03:34 2016

@author: Giulero"""

import random
from class_work import HW1

img_folder = r"../coil-100/"

#Choose random value of images
n_image=[]
while len(n_image)<4:
#    n_image.append(int(random.random()*100))
    n_image.append(random.randrange(1,99))
n_image.sort()

hw = HW1(img_folder, n_image) #Initialize object
hw.X_() #Create features Matrix
hw.Y() #Create label vector
hw.PCA(2) #Fit pca
hw.split_data() #Split data in train and test data
hw.fit() #Fit classifier
hw.plt()
#hw.covariance_plot()
hw.PCA(4) #Fit pca
hw.split_data() #Split data in train and test data
hw.fit() #Fit classifier
hw.plt()
#hw.covariance_plot()
hw.PCA(11) #Fit pca
hw.split_data() #Split data in train and test data
hw.fit() #Fit classifier
hw.plt()
hw.covariance_plot()
