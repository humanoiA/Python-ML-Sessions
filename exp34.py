import numpy as np
import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
x=[[5,55,44,15],[98,6,7,72],[56,8,12,38],[33,18,11,31],[12,76,98,28],[12,34,42,18],[2,104,120,9],[13,54,37,19]]
y=['A','U/A','U/A','U/A','B','A','C','A']
p=[[40,100,110,90]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
clf=DecisionTreeClassifier()
clf=clf.fit(x,y)
print(clf.predict(p))
print('Accuracy',clf.score(x_train,y_train))