from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA

print("Importing -> [numpy -glob -plt -PCA]")






imageFolderPath = 'C:\Users\the_s\OneDrive\Documenti\Python\ML_Hw1\coil-100'
imagePath = glob.glob( imageFolderPath+'\obj1_*')
img_data = np.array([np.array(Image.open(imagePath[i]).convert('L'),'f') for i in range(len(imagePath))])

print("Image_Loaded")

X = img_data.ravel()
y = X
X = np.array(X).reshape((24,-1))
print(X.shape)

X_t = PCA(2).fit_transform(X)
plot.scatter(X_t[:, 0], X_t[:, 1], c=y)
#plot.scatter(X_t[:, 0], X_t[:, 1],)
