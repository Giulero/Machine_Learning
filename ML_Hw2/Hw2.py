# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:48:24 2017

@author: the_s
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

print('----- Load Data -----')
X_train = numpy.load(r'regression_Xtrain.npy')
Y_train = numpy.load(r'regression_ytrain.npy')
X_test = numpy.load(r'regression_Xtest.npy')
Y_test = numpy.load(r'regression_ytest.npy')

plt.scatter(X_train, Y_train, c='r', label='Train_data')
plt.scatter(X_test, Y_test, c='b', label='Test_data')
plt.legend(loc='upper left')
plt.savefig('scatterPlot.pdf', format='pdf')
plt.show()

#lr=linear_model.LinearRegression()
#lr.fit(X_train.reshape(-1,1),Y_train)
#plt.plot(X_test, lr.predict(X_test.reshape(-1,1)), label='Linear Model')
#plt.scatter(X_test, Y_test, label='Original data')
#plt.legend(loc='upper left')
#plt.show()
#
#predictions=lr.predict(X_test.reshape(-1,1))
#mean_square_e=mean_squared_error(Y_test, predictions)
#print('Mean square error = ', mean_square_e)

mean_square_error = []
poly_grade = range(1,10)
min_msq = 2000 #mean_square_e
n=1
for i in poly_grade:
    poly=PolynomialFeatures(degree=i, include_bias=False)
    xPoly=poly.fit_transform(X_train.reshape(-1,1))
    lr=linear_model.LinearRegression()
    lr.fit(xPoly, Y_train) #xPoly fa sempre riferimento a X_train!
    #x_range=np.linspace(-1, 5.5, 100)
    X_test_i=poly.fit_transform(X_test.reshape(-1,1))
    predicted = lr.predict(X_test_i)
    plt.figure(i)
    plt.plot(X_test, predicted, c='k', label='Polynomial degree = '+str(i))
    plt.scatter(X_train, Y_train, c='b', label="Train_data")
    plt.scatter(X_test, Y_test, c='r', label='Test_data')
    plt.legend(loc='upper left')
    msq=mean_squared_error(Y_test, predicted)
    plt.title('-- Degree %d -- \n-- MSE %.2e --' % (i, msq))
    mean_square_error.append(msq)
    plt.savefig('Degree'+str(i)+'.pdf', format='pdf')
    plt.show()
    print('Mean square error with polynomial of grade ', i)
    print('----- ', msq, ' -----')
    plt.scatter(predicted, predicted-Y_test)
    plt.axis('equal')
    plt.ylim(-20, 20)
    plt.xlim(-10, 10)
    plt.hlines(y=0, xmin=-20, xmax=20, lw=2, color='red', label='Zero error line')
    plt.legend(loc='upper left')
    plt.title('Residuals\n Degree %d ' % i)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.savefig('Residuals'+str(i)+'.pdf', format='pdf')
    plt.show()
    if msq < min_msq :
        min_msq = msq
        n = i
plt.plot(poly_grade, mean_square_error, marker = 'o')
plt.title('-- Mean square error --')
plt.savefig('MeanSquareError.pdf', format='pdf')
plt.show()
print('----- Best polynomial choice -----')
print('Minimum Square error \t=\t', min_msq)
print("Grade of polynomial \t=\t", n)
