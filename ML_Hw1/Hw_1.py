#Homework No.1

from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from decisionboundary import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import random



img_data = []
img_data_raveled = []
img_folder = r'coil-100/'
n_image=[]
while len(n_image)<4:
    n_image.append(int(random.random()*100))
img_list=[]
for num in n_image:
    img_list+=glob.glob(img_folder+'obj'+str(num)+'_*')
#img_list = glob.glob(img_folder+"\obj14_*")+glob.glob(img_folder+"\obj16_*")+glob.glob(img_folder+"\obj17_*")+glob.glob(img_folder+"\obj18_*")
for filename in img_list:
    im = np.asarray(Image.open(filename).convert("RGB"))
    im_raveled = np.ravel(im)
    img_data_raveled.append(im_raveled)

#def import_images(img_folder, list):
#    img_folder_=img_folder+"\obj"
#    for num in list:
#        img_list.append(img_folder_+num2str(list(num)))+"_*"
#        y.append([num]*len(img_list)/len(list))
#    for filename in img_list:
#        img_data_raveled.append(np.ravel(np.asarray(Image.open(filename).convert("RGB"))))
#    X = np.array(img_data_raveled).reshape((len(img_list)), -1)
#

X=[]
X = np.array(img_data_raveled).reshape((len(img_list)), -1)
#y = [1]*72+[2]*72+[3]*72+[4]*72
y = []
for num in n_image:
    y += [num]*int(len(img_list)/len(n_image))
y.sort()

sc = StandardScaler()
pca = PCA(2)
#X_scaled = preprocessing.scale(X)
X_std = sc.fit_transform(X)
X_std_pca = pca.fit_transform(X_std)
plt.scatter(X_std_pca[0:72,0], X_std_pca[0:72,1], c='y')
plt.scatter(X_std_pca[72:144,0], X_std_pca[72:144,1], c='m')
plt.scatter(X_std_pca[144:216,0], X_std_pca[144:216,1], c='r')
plt.scatter(X_std_pca[216:288,0], X_std_pca[216:288,1], c='g')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_std_pca, y, test_size=0.5, random_state=55)

clf = GaussianNB()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print ("accuracy: ", accuracy)


markers = ('o', 'x', 's', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))]) #unique unisce gli elementi uguali
#plot the decison surface
x1_min, x1_max = X_std_pca[:,0].min() - 1, X_std_pca[:,0].max()+1
x2_min, x2_max = X_std_pca[:,1].min() - 1, X_std_pca[:,1].max()+1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
#crea una meshgrid con valori tra x_min e x_max con data risoluzione. arange crea un vettore tra x_min e x_max con passo resolution
Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #.T->Transpose. Ravel mette in riga gli elementi
Z = Z.reshape(xx1.shape) #shape ritorna la dimensione della matrice/vettore. Reshape modella una matrice/vettore dando le dimensioni desiderate
plt.contourf(xx1, xx2, Z, alpha=0.4)
for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X_std_pca[y == cl, 0], y=X_std_pca[y == cl, 1], alpha=0.8, c=cmap(idx), label=cl)

plt.legend(loc='upper left')


#Plot the decision boundary. For that, we will assign a color to each
#point in the mesh [x_min, x_max]x[y_min, y_max].
#h = 0.2 #step size in the mesh
#x_min, x_max = X_t[:, 0].min() - 1, X_t[:, 0].max() + 1
#y_min, y_max = X_t[:, 1].min() - 1, X_t[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#np.arange(y_min, y_max, h))
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
## Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#plt.figure()
plt.show()
