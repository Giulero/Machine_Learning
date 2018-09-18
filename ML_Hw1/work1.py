import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PIL import Image
import numpy as np
import glob
import random
from classificator import *



class work1:

	name = ""
	list_chosen_class	= [] 			 #List of the chosen class
	list_images_path	= []			 #List of the images path
	list_images_matrix	= np.asarray([]) #List of the images matrix	
	list_lable_class	= []
	X = []
	X_t = []

	static_path = r'C:\Users\the_s\OneDrive\Documenti\Python\ML_Hw1\coil-100'
	obj 		=  "\obj"
	obj_end 	= "__*"


	#Constructor 
	def __init__(self,Name):
		self.name = ""
		self.list_chosen_class	= [] 			 
		self.list_images_path	= []			 
		self.list_images_matrix	= np.asarray([]) 
		self.list_lable_class	= []
		self.X = []
		self.X_t = []
		self.list_chosen_class = [1,2,3,4]
		self.name = Name
		print("########  "+self.name+"  ########")
		#hw1.random_class()
		self.load_all_image_path()
		self.load_all_class_label() 
		self.load_images_matrix()



	#Choose 4 randome class 
	def random_class(self):
		self.list_chosen_class = [self.rand_num(),self.rand_num(),self.rand_num(),self.rand_num()]

	#Add single image path to < list_images_path >
	def load_single_image_path(self,num_obj):
		temp_path = self.obj + str(self.list_chosen_class[num_obj]) + self.obj_end
		print("Path loaded - Class "+str(self.list_chosen_class[num_obj])+" .")
		self.list_images_path.extend(glob.glob(self.static_path+temp_path))


	#Call  n = len(self.list_chosen_class) numero delle classi scelte add_single_image_path
	def load_all_image_path(self):
		for obj_num in range(0,len(self.list_chosen_class)):
			self.load_single_image_path(obj_num)



	#Load images into < list_images_matrix >
	def load_images_matrix(self):
		self.list_images_matrix = np.asarray([np.asarray(Image.open(self.list_images_path[i]).convert('L'), 'f') for i in range(len(self.list_images_path))])
		print("Image matrix loaded - list_images_matrix len = "+str(len(self.list_images_matrix))+" .")



	#Append 72 times the vlaue to < list_lable_class >
	def load_single_class_label(self,value):
			for i in range(0,72): 
				self.list_lable_class.append(value)



	#Call for the number of class chosen load_single_class_label
	def load_all_class_label(self):
		for obj in range(1,len(self.list_chosen_class)+1):
			self.load_single_class_label(obj)


	#Give random value betwen 1 - 99
	def rand_num(self):
		return random.randrange(1,99)

	#Return X_t
	def get_X(self):
		return self.X_t

	#Return Y_t
	def get_Y(self):
		return self.list_lable_class

	#ravel create from a (288,128,128) to matrix a (4718592,) vector
	#model X and create from (4718592,) vector a (288, 16384) 
	#unity-variance #zero-mean
	def X_value(self):
		self.X = self.list_images_matrix.ravel()  	
		self.X = np.array(self.X).reshape(288, -1)		
		self.X = preprocessing.scale(self.X)				

	# PCA(2) compute the first and second PCA vectors, fit.transform(X)
	def pca(self,value):
		self.X_t = PCA(value).fit_transform(self.X)				

	#Plot with scatter the X_t[:, value1],X_t[:, value2]
	def plot(self,value1,value2):
		plt.scatter(self.X_t[:, value1], self.X_t[:, value2] ,c=self.list_lable_class)		#plot
		plt.show()


	def get_only_coloum_X_t(self,colum1,colum2):
		self.X_t = self.X_t[:,[colum1,colum2]]






	





