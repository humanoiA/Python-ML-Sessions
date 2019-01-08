import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pa
iris=datasets.load_iris()
x,y=iris.data,iris.target
kmeans=KMeans(n_clusters=3)
kmodel=kmeans.fit(x)
centroid=kmeans.cluster_centers_
print(y)
print(kmodel.labels_)
print(kmodel.cluster_centers_)
print(pa.crosstab(y,kmodel.labels_))
labels=kmeans.labels_
color=['g.','r.','c.','y.']
for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],color[labels[i]],markersize=20)
    plt.scatter(centroid[:,0],centroid[:,1],marker='x',zorder=10,s=150,linewidths=5)
plt.show()