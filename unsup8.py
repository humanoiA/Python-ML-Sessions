import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
df=pd.read_excel('data.xlsx')
df1=df.drop(['ID Tag','Model','Department'],axis=1).values
#print(df1)
km=KMeans(n_clusters=4,n_init=4)
x=km.fit_predict(df1)
labels=km.labels_
#print(len(labels))
cluster_center=km.cluster_centers_
print(cluster_center)
fig=plt.figure(figsize=(4,3))
plt.clf()
ax=Axes3D(fig,rect=[0,0,1,1],elev=48)
plt.cla
#print(type(labels))
ax.scatter(df1[:,6],df1[:,0],df1[:,1],c=labels.astype(np.float))
ax.w_xaxis.set_ticklabels(km.cluster_centers_[:,0])
ax.w_yaxis.set_ticklabels(km.cluster_centers_[:,1])
ax.w_zaxis.set_ticklabels(km.cluster_centers_[:,2])
ax.set_xlabel('Travel Cost')
ax.set_ylabel('Work Order')
ax.set_zlabel('Avg. Cost')
plt.show()
