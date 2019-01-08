import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
filedata='fruit_data_with_colors.txt'
name=['mass','width','height','color_score']
fdata=pd.read_table(filedata)
x=fdata[name]
y=fdata['fruit_label']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
model1=GaussianNB()
model2=LinearDiscriminantAnalysis()
model3=SVC()
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
p=[[162,7.4,7.2,0.85]]
pred1=model1.predict(x_test)
pred2=model2.predict(p)
pred3=model3.predict(p)
#print(LDA.score(x_train,y_train))
#print(LDA.score(x_test,y_test))
print(pred1,pred2,pred3)
#np.reshape(pred1)
print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))
#print(confusion_matrix(y_test,pred2))
#print(classification_report(y_test,pred2))
#print(confusion_matrix(y_test,pred3))
#print(classification_report(y_test,pred3))