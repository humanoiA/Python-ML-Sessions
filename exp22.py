from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(url,names=name)
array = data.values
x = array[:,0:8]
y = array[:,8]
model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
