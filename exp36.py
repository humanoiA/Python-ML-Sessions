from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
fdata=pd.read_table("fruit_data_with_colors.txt")
feature_names=['mass','width','height','color_score']
x=fdata[feature_names]
y=fdata['fruit_label']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train))
print(knn.score(x_test,y_test))
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))