import numpy as np
import pandas as pd
from pylab import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
x=[[190,70,44],[166,65,45],[190,90,47],[175,64,39],[171,75,40],[177,80,42],[160,60,83],[144,54,37]]
y=['male','male','male','male','female','female','female','female']
p=[[162,95,44]]
clf=DecisionTreeClassifier()
clf=clf.fit(x,y)
print(clf.predict(p))
clf1=KNeighborsClassifier()
clf1=clf1.fit(x,y)
print(clf1.predict(p))
clf2=MLPClassifier()
clf2=clf2.fit(x,y)
print(clf2.predict(p))
clf3=RandomForestClassifier()
clf3=clf3.fit(x,y)
print(clf3.predict(p))