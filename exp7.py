import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
url="indians-diabetes.data"
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pa.read_csv(url,names=name)
#data.hist()
#plt.show()
data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
plt.show()