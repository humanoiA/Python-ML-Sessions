from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(url,names=name)
array = data.values
x = array[:,0:8]
y = array[:,8]
model=LogisticRegression()
rfe=RFE(model,3)
fit=rfe.fit(x,y)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)