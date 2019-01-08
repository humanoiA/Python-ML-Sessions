from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(url,names=name)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]
num_folds=10
test_size=0.33
seed=7
kfold = ShuffleSplit(n_splits=num_folds,test_size=test_size,random_state=seed)
model = LogisticRegression()
results=cross_val_score(model,x,y,cv=kfold)
print(results.mean()*100.0,results.std()*100.0)