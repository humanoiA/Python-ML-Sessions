from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
fruit = pd.read_table("fruit_data_with_colors.txt")
f_names = ['mass','width','height','color_score']
x=fruit[f_names]
y=fruit['fruit_label']
print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
logreg=DecisionTreeClassifier()
logreg=logreg.fit(x_train,y_train)
print('Accuracy',logreg.score(x_train,y_train))
print('Accuracy',logreg.score(x_test,y_test))