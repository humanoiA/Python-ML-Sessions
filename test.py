import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
dt=pd.read_csv('Tanis.csv')
#print(dt)
#b=pd.DataFrame(load_boston())
#print(b.head(50))
number=LabelEncoder()
dt['outlook']=number.fit_transform(dt['outlook'])
dt['temperature']=number.fit_transform(dt['temperature'])
dt['humidity']=number.fit_transform(dt['humidity'])
dt['wind']=number.fit_transform(dt['wind'])
dt['playtennis']=number.fit_transform(dt['playtennis'])
#print(dt)
g=[['overcast','cool','normal','strong']]
g[0]=number.fit_transform(g[0])
arr=dt.values
x=arr[:,0:4]
y=arr[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
model=LinearDiscriminantAnalysis()
p=model.fit(x_train,y_train)
pred=model.predict(g)
print(pred)