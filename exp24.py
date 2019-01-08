from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(url,names=name)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]
num_folds=10
loocv=LeaveOneOut()
#kfold = KFold(n_splits=num_folds,random_state=seed)
model = LogisticRegression()
results=cross_val_score(model,x,y,cv=loocv)
print(results.mean()*100.0,results.std()*100.0)