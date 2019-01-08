from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
label_name=data['target_names']
feature_name=['feature_names']
feature=data['data']
labels=data['target']
print(label_name,feature_name,labels)
x_train,x_test,y_train,y_test=train_test_split(feature,labels,test_size=0.33,random_state=12)
nb=GaussianNB()
model=nb.fit(x_train,y_train)
pred=model.predict(x_test)
print(accuracy_score(y_test,pred))