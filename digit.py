import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
data=pd.read_csv("train.csv").values
print(len(data))
clf=DecisionTreeClassifier()
x_train=data[0:21000,1:]
train_lable=data[21000:,0]
print(train_lable)
clf.fit(x_train,train_lable)
x_test=data[21000:,1:]
act_label=data[21000:,0]
d=x_test[8]
d.shape=(28,28)
plt.imshow(d,cmap='gray')
plt.show()
print(clf.predict(x_test[[8]]))