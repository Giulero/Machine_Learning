# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:24:53 2017

@author: the_s
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from logistic_regression import LinearRegressionGD
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep ='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'CRIM', 'B', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()

sns.reset_orig()
#X = df[['RM']].values
X = df.iloc[:, :-1].values #aggiunto
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #aggiunto     
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X_std = sc_x.fit_transform(X)
#y_std = sc_y.fit_transform(y)
#lr = LinearRegressionGD()
lr = LinearRegression() #modificato
lr.fit(X_train, y_train) #modificato
#plt.plot(range(1, lr.n_iter+1), lr.cost_)
y_train_pred = lr.predict(X_train) #aggiunto
y_test_pred = lr.predict(X_test) #aggiunto
plt.scatter(y_train_pred, y_train_pred -y_train, c='blue', marker='o', label='Trainig data') #aggiunto
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data') #aggiunto
plt.ylabel('Residuals') #modificato
plt.xlabel('Predicted values') #modificato
plt.legend(loc='upper left') #aggiunto
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red') #aggiunto
plt.xlim([-10, 50]) #aggiunto
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

#lin_regplot(X_std, y_std, lr)
#plt.xlabel('Average numbers of room [RM] (standardized)')
#plt.ylabel('Price in $1000"s [MEDV] (standardized)')
#plt.show()

X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

#create polynomial features
quadratic = PolynomialFeatures(degree=2)      
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

#linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

#quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

#cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

#plot results
plt.figure()
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2,linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2,linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2,linestyle='--')
plt.xlabel('Lower status of the population $[LSTAT]$')
plt.ylabel('Price in $1000$ $[MEDV]$')
plt.legend(loc='upper right')
plt.show()

#transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

#fit features
X_fit = np.arange(X_log.min()-1, X_log.max()+1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

#plt resuluts
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' %linear_r2, lw=2)
plt.xlabel('log(Lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
plt.legend(loc='upper left')
plt.show()
