from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(url,names=name)
array = data.values
x = array[:,0:8]
y = array[:,8]
test = SelectKBest(score_func=chi2,k=2)
fit = test.fit(x,y)
set_printoptions(precision=3)
print(fit.scores_)
features= fit.transform(x)
print(features[0:5,:])
