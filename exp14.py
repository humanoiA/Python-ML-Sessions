import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
data=pd.read_csv('E:\\ML stuff\\gender_refine-csv.csv')
print(data['gender'].unique())
d=data.drop(['score'],axis=1)
array=d.values
cv=CountVectorizer()
#x=cv.fit_transform(array[:,0])
#print(d.dtypes)
x=d['name']
y=d['gender']
a=cv.fit_transform(x)
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
x_train,x_test,y_train,y_test=train_test_split(a,y,test_size=0.20,random_state=0)
model.fit(x_train,y_train)
#print(model.score(x_test,y_test))
vary=[input()]
vect=cv.transform(vary).toarray()
if model.predict(vect)==0:
    print('Female')
else:
    print('Male')
print('Accuracy',model.score(x_train,y_train))
print('Accuracy',model.score(x_test,y_test))
