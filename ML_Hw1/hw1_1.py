
from work1 import * 
from classificator import *


#Data preparation [1 - 2] 
hw1 = work1("")
#3 

#Principal Component Analisis
#1
hw1.X_value()
#2
hw1.pca(2)
#3
hw1.plot(0,1)

#1
hw2 = work1("Second Test")
hw2.X_value()
hw2.pca(5)
hw2.plot(3,4)

#1
hw3 = work1("Thitd Test")
hw3.X_value()
hw3.pca(12)
hw3.plot(10,11)

#CLASSIFICATOR 

cl1 = classificator(hw1.get_X(),hw1.get_Y(),"First Test")
cl1.train_test()
cl1.gaussianNB()
cl1.plot()
cl1.accuracy()

hw2.get_only_coloum_X_t(3,4)
cl2 = classificator(hw2.get_X(),hw2.get_Y(),"Second Test")
cl2.train_test()
cl2.gaussianNB()
cl2.plot()
cl2.accuracy()


hw3.get_only_coloum_X_t(10,11)
cl3 = classificator(hw3.get_X(),hw3.get_Y(),"Second Test")
cl3.train_test()
cl3.gaussianNB()
cl3.plot()
cl3.accuracy()