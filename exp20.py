import  numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model
d=pd.read_csv('stud1.csv')
pt.scatter(d[['AGE']], d['RESULT'],color='orange',label='scatter')
pt.show()
clf=linear_model.LinearRegression()
clf=clf.fit(d[['AGE']], d['RESULT'])
result=clf.predict([[18],[22]])
pt.scatter(d[['AGE']], d['RESULT'],color='orange',label='scatter')
pt.plot([[18],[22]],result,color='blue')
pt.xlabel('AGE')
pt.ylabel('CGPA')
pt.legend(loc='lower right')
pt.title('CGPA PREDICTION ACCORDING TO ROLL')
print('Result',result)
pt.show()
