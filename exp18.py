import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split
boston=datasets.load_boston(return_X_y=False)
x=boston.data
y=boston.target
#print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.coef_)
print(reg.predict(x_test))
v_score=reg.score(x_test,y_test)
print(v_score)
plt.scatter(reg.predict(x_train),reg.predict(x_train)-y_train,color='r',s=10,label='Train Data')
plt.scatter(reg.predict(x_test),reg.predict(x_test)-y_test,color='b',s=10,label='Test Data')
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)
plt.title('First Demo')
plt.legend(loc='upper right')
plt.show()