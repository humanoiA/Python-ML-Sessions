    from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe=read_csv(url,names=name)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)
model = LogisticRegression()
scoring = 'neg_log_loss'
results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
print(results.mean(),results.std())
