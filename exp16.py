from pandas import read_csv
from sklearn.decomposition import PCA
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(url,names=name)
array = data.values
x = array[:,0:8]
y = array[:,8]
pca = PCA(n_components=3)
fit=pca.fit(x)
print(fit.explained_variance_ratio_)
print(fit.components_)
