import exp32
from numpy import *
dataset,label=exp32.createDataSet()
print(dataset,label)
p=array([[2.0,2.1]])
result=exp32.classify(p,dataset,label,3)
print(p,result)