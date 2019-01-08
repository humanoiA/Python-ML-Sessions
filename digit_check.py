import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from scipy.misc import imread,bytescale,imresize
digit=pd.read_csv("train.csv").values
x=digit[0:21000,1:]
y=digit[21000:,0]
print(x.dtype)
print(x.shape)
print(x.max())
print(x.min())
clf=svm.SVC(gamma=0.1)
clf.fit(x,y)
img=imread('Chars/5.jpg')
img=imresize(img,(8,8))
img=img.astype(x.dtype)
img=bytescale(img,high=16,low=0)
x_testdata=[]
for r in img:
    for c in r:
        x_testdata.append(sum(c)/3.0)
data=clf.predict([[x_testdata]])
print(data)

