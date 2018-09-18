# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:49:42 2016

@author: the_s
"""

import numpy as np
from PIL import Image

img = Image.open(r'C:\Users\the_s\OneDrive\Documenti\Python\ML_Hw1\coil-100\obj1__35.png').convert('RGBA')
arr = np.array(img)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')
img2.show()