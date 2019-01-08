import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
url='indians-diabetes.data'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pd.read_csv(url,names=name)
array=data.values
X=array[:,0:8]
Y=array[:,8]
scalar=StandardScaler().fit(X)
rescaleX=scalar.transform(X)
np.set_printoptions(precision=3)
print(rescaleX[0:5,:])
