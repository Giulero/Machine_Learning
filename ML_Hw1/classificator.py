import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


class classificator:

	name = ""
	X = 0
	Y = 0
	X_train =0
	X_test  =0
	Y_train =0
	Y_test  =0

	clf = []
	prediction = []	

	def __init__(self,X,Y,name):
		print("\n******  "+name+"  ******")
		self.X = X
		self.Y = Y
		self.name = name
		self.clf = GaussianNB()

	def train_test(self):
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.50,random_state=0)		#Divide in 2 porzioni da 144,3 mischiate 

	def gaussianNB(self):
		self.clf.fit(self.X_train,self.Y_train)	              #Fit gaussian naive bayes according X,Y.
		self.prediction = self.clf.predict(self.X_test)	
		self.accuracy()

	def accuracy(self):
		score = self.clf.score(self.X_test,self.Y_test)
		print("Accuracy of  "+ str(score*100) +" %  .")

	
	def plot_select_coloum(self,value1,value2):
		plt.scatter(self.X_test[:, value1], self.X_test[:, value2] ,c=self.prediction)		#plot
		plt.show()

	def plot(self):
		plt.scatter(self.X_test[:, 0], self.X_test[:, 1] ,c=self.prediction)		#plot
		plt.show()


	
		


	




