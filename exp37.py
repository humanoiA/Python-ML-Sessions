from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
data = load_iris()
label_name=data['target_names']
feature_name=['feature_names']
feature=data['data']
labels=data['target']
#print(label_name,feature_name,labels)
x_train,x_test,y_train,y_test=train_test_split(feature,labels,test_size=0.33,random_state=12)
nb=GaussianNB()
model=nb.fit(x_train,y_train)
pred=model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))