import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
url='indians-diabetes.data'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pandas.read_csv(url,names=name)
scatter_matrix(data)
plt.show()