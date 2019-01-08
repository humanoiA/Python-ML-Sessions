from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
training_accuracy=[]
testing_accuracy=[]
neighbours_setting=range(1,11)
for n in neighbours_setting:
    clf=KNeighborsClassifier(n_neighbors=n)
    clf.fit(x_train,y_train)
    testing_accuracy.append(clf.score(x_test,y_test))
    training_accuracy.append(clf.score(x_train,y_train))
plt.plot(neighbours_setting,training_accuracy,label='Training Accuracy')
plt.plot(neighbours_setting,testing_accuracy,label='Testing Accuracy')
plt.legend()
plt.show()
