# -*- coding: utf-8 -*-

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from decisionboundary import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_std = preprocessing.scale(X)
X_pca = PCA(2).fit_transform(X_std)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.4) #,random_state=0)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

acc_k = []
acc_k_d = []

for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test) #accuracy_score(y_test, pred)
    plot_decision_regions(X_combined, y_combined, classifier=knn)#, test_idx=range(len(X_pca)-len(X_test), len(X_pca)))
#    plt.xlabel('petal lenght [std]')
#    plt.ylabel('petal width [std]')
    plt.legend(loc='upper left')
    plt.title('k = %d \n --- Accuracy = %f ---' % (k, accuracy_score(y_test, pred)))
    plt.savefig('Images/k_'+str(k)+'.pdf', format='pdf')
    acc_k.append(accuracy_score(y_test, pred))
    plt.show()
    
plt.plot(range(1,11), acc_k, marker='o')
plt.xlabel('K-neighbors')
plt.ylabel('Accuracy')
plt.ylim(0.83, 0.96)
plt.savefig('Images/Accuracy_k.pdf', format='pdf')
plt.show()
    
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test) #accuracy_score(y_test, pred)
    plot_decision_regions(X_combined, y_combined, classifier=knn)#, test_idx=range(len(X_pca)-len(X_test), len(X_pca)))
#    plt.xlabel('petal lenght [std]')
#    plt.ylabel('petal width [std]')
    plt.legend(loc='upper left')
    plt.title('k = %d \n --- Accuracy = %f ---' % (k, accuracy_score(y_test, pred)))
    plt.savefig('Images/k_distance'+str(k)+'.pdf', format='pdf')
    acc_k_d.append(accuracy_score(y_test, pred))
    plt.show()
    
plt.plot(range(1,11), acc_k_d, marker='o')
plt.xlabel('K-neighbors')
plt.ylabel('Accuracy')
plt.ylim(0.83, 0.96)
plt.savefig('Images/Accuracy_k_distance.pdf', format='pdf')
plt.show()

    
for x in ['uniform', 'distance']:
    knn1 = KNeighborsClassifier(n_neighbors=3, weights = x)
    knn1.fit(X_train, y_train)
    accuracy = knn1.score(X_test, y_test) #accuracy_score(y_test, pred)
    plot_decision_regions(X_combined, y_combined, classifier=knn1)#, test_idx=range(len(X_pca)-len(X_test), len(X_pca)))
#    plt.xlabel('petal lenght [std]')
#    plt.ylabel('petal width [std]')
    plt.legend(loc='upper left')
    plt.title('Class classification(k = %d, weights = "%s") \n --- Accuracy = %f ---' % (3, x, accuracy_score(y_test, pred)))
    plt.savefig('Images/k3_'+x+'.pdf', format='pdf')
    plt.show()

acc_g = []

for alpha in [0.1, 1, 10, 100, 1000]:
    def custom_weights(dist):
        return np.exp(-alpha*dist**2)
    knn2 = KNeighborsClassifier(n_neighbors=7, weights=custom_weights)
    knn2.fit(X_train, y_train)
    pred = knn2.predict(X_test)
    print(knn2.score(X_test, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=knn2)
    plt.legend(loc='upper left')
    plt.title('k = 7 -- Gaussian(alpha=%0.1f) \n Accuracy = %f' % (alpha, accuracy_score(y_test, pred)))
    plt.savefig('Images/alpha_'+str(alpha)+'.pdf', format='pdf')
    acc_g.append(accuracy_score(y_test, pred))
    plt.show()
    
plt.plot([0.1, 1, 10, 100, 1000], acc_g, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
#plt.ylim(0.84, 0.96)
plt.savefig('Images/Accuracy_gaussian.pdf', format='pdf')
plt.show()

#Best solution
accur_max = 0
k_max = 0
weight_max = ''
alpha_max = 0
for k in range(1,11):
    for w in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=k, weights=w)
        knn.fit(X_train, y_train)
        accur = knn.score(X_test, y_test)
        print('k = %d -- weights = %s \t -- \t Accuracy = %f' % (k, w, accur))
        if accur > accur_max:
            accur_max = accur
            k_max = k
            weight_max = w
            knn_best = knn
    for alpha in [0.1, 1, 10, 100, 1000]:
        def custom_weights(dist):
            return np.exp(-alpha*dist**2)
        knn = KNeighborsClassifier(n_neighbors=k, weights=custom_weights)
        knn.fit(X_train, y_train)
        accur = (knn.score(X_test, y_test))
        print('k = %d -- weights = gaussian(alpha=%s) \t -- \t Accuracy = %f' % (k, alpha, accur))
        if accur > accur_max:
            accur_max = accur
            k_max = k
            alpha_max = alpha
            weight_max = 'Gaussian(alpha='+str(alpha)+')'
            knn_best = knn 
            
best_choice = [k_max, weight_max, accur_max]


print('Best choice is k = %d, weights = %s \nGives accuracy equal to %f' % (k_max, weight_max, accur_max))
#knn_best=KNeighborsClassifier(n_neighbors=k_max, weights=weight_max)
knn_best.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=knn_best)
plt.legend(loc='upper left')
plt.title(' k = %d -- Weights : %s  \nAccuracy = %f' % (k_max, weight_max, accur_max))
plt.savefig('Images/Best_choice_'+weight_max+'.pdf', format='pdf')
plt.show()            