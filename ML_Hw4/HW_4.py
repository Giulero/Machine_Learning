# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:30:54 2017

@author: Giulero
"""

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from matplotlib.colors import Normalize
#import seaborn as sns
from decisionboundary import plot_decision_regions

def evaluate_linear_SVM(param, X_tst, y_tst):
    svm = SVC(kernel='linear', C=param)
    svm.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=svm)
    plt.xlabel('Sepal lenght [cm]')
    plt.ylabel('Sepal width [cm]')
    plt.legend(loc='upper left')
    plt.title('SVM-linear -- C=%0.3f' % (param))
    print('Accuracy = %f' %(svm.score(X_tst, y_tst)))
    plt.savefig('lin_C'+str(param)+'.pdf')
    plt.show()
    return svm.score(X_tst, y_tst)

def evaluate_rbf_SVM(param, gam, X_tst, y_tst):
    svm = SVC(kernel='rbf', C=param, gamma=gam)
    svm.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=svm)
    plt.xlabel('Sepal lenght [cm]')
    plt.ylabel('Sepal width [cm]')
    plt.legend(loc='upper left')
    plt.title('SVM-RBF -- C=%s -- gamma=%s' % (param, gam))
    plt.savefig('rbf_C'+str(param)+'G'+str(gam)+'.pdf')  
    plt.show()
    print('Accuracy = %f' %(svm.score(X_tst, y_tst)))
    return svm.score(X_tst, y_tst)

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

iris = datasets.load_iris()
X = iris.data[:, [0,1]]
y = iris.target
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.4)
X_combined = np.vstack((X_train, X_val))
y_combined = np.hstack((y_train, y_val))

C_range = np.logspace(-3, 3, 7) #Passa da 10 a 13 dopo
gamma_range = np.logspace(-5, 5, 6)
acc_l = []
for c in C_range:
    #Plotta e mette in lista i valori di accuratezza nella lista 'acc'
    acc_l.append(evaluate_linear_SVM(c, X_val, y_val))

plt.plot(C_range, acc_l, marker='o')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('C')
#plt.xticks(C_range)
plt.savefig('c_accuracy.pdf', format='pdf')
plt.show()

for c,a in zip(C_range, acc_l):
    print('C = %s \t Accuracy = %f' % (c, a))

max_acc = max(acc_l)
best_index = acc_l.index(max_acc)
best_accuracy = evaluate_linear_SVM(C_range[best_index], X_test, y_test)

if best_accuracy < max_acc:
    print('Accuracy on test data is worse than validation data')
else:
    print('Good choice!')


acc_rbf_c =[]
for c in C_range:
    acc_rbf_c.append(evaluate_rbf_SVM(c, 'auto', X_val, y_val))

plt.plot(C_range, acc_rbf_c, marker='o')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.savefig('c_accuracy_rbf_c.pdf', format='pdf')
plt.show()    

max_acc_rbf_c = max(acc_rbf_c)
best_index_rbf_c = acc_rbf_c.index(max_acc_rbf_c)
best_accuracy_rbf_c = evaluate_rbf_SVM(C_range[best_index_rbf_c],'auto', X_test, y_test)

#grid search
scores = np.empty((len(C_range), len(gamma_range)))
for i, c in enumerate(C_range):
    for j,gamma in enumerate(gamma_range):
        scores[i,j]=evaluate_rbf_SVM(c, gamma, X_val, y_val)
print(scores)
np.savetxt('scores.csv', scores)

max_acc_rbf=scores.max()
max_index_rbf=np.where(scores == scores.max())
print('max accuracy = %s -- C=%s gamma=%s' % (max_acc_rbf, C_range[max_index_rbf[0]], gamma_range[max_index_rbf[1]]))

best_accuracy_rbf=evaluate_rbf_SVM(C_range[max_index_rbf[0]].max(), gamma_range[max_index_rbf[1]].max(), X_test, y_test)

#hm = sns.heatmap(scores, cbar=True, annot=False, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=gamma_range, xticklabels=C_range)
#plt.xlabel('gamma')
#plt.ylabel('C')
#plt.title('Validation accuracy')
#plt.savefig('validation_accuracy_1.pdf')
#plt.show()
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.6, midpoint=0.92)) #plt.cm.rainbow
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.savefig('validation_accuracy.pdf', format='pdf', bbox_inches='tight')
plt.show()

X_folds = np.array_split(X_combined, 5)
y_folds = np.array_split(y_combined, 5)

scores_k = np.empty((len(C_range), len(gamma_range)))
scor = []
for i, c in enumerate(C_range):
    for j, gamma in enumerate(gamma_range):
        for k in range(5):
            X_tr = list(X_folds)
            X_val = X_tr.pop(k)
            X_tr = np.concatenate(X_tr)
            y_tr = list(y_folds)
            y_val = y_tr.pop(k)
            y_tr = np.concatenate(y_tr)
            scor.append(evaluate_rbf_SVM(c, gamma, X_val, y_val))
        scores_k[i,j] = np.mean(scor)
        scor = []
print(scores_k)
np.savetxt('scores_k.csv', scores_k)     

max_acc_k=scores_k.max()
max_index_k=np.where(scores_k == scores_k.max())
print('max accuracy = %s -- C=%s gamma=%s' % (max_acc_k, C_range[max_index_k[0]], gamma_range[max_index_k[1]]))

best_accuracy_k=evaluate_rbf_SVM(C_range[max_index_k[0]].max(), gamma_range[max_index_k[1]].max(), X_test, y_test)       

plt.imshow(scores_k, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=0.5, midpoint=0.92)) #plt.cm.rainbow
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy -- k fold')
plt.savefig('validation_accuracy_k.pdf', format='pdf', bbox_inches='tight')
plt.show()