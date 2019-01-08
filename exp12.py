import pandas
import numpy
import matplotlib.pyplot as plt
url='indians-diabetes.data'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pandas.read_csv(url,names=name)
correlations = data.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(name)
ax.set_yticklabels(name)
plt.show()