from sklearn import tree
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])
rm=[0,50,100]
new_target=np.delete(iris.target,rm)
new_data=np.delete(iris.data,rm,axis=0)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(new_data,new_target)
p=clf.predict([[5.0,3.1,1.7,0.4]])
print('Original data',iris.target[rm])
print('Predicted Data',p)
print('Accuracy',model.score(x_test,y_test))