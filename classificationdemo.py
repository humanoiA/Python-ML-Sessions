import d2
from numpy import *
dataset,label=d2.createDataSet()
print(dataset,label)
p=array([[0.1,.0]])
result=d2.classify(p,dataset,label,3)
print(p,result)