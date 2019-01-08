import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

x=[1,5,1.5,8,1,9]
y=[2,8,1.8,8,0.6,11]
plt.scatter(x,y)
plt.show()
X=np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
centroid=kmeans.cluster_centers_
print(centroid)
labels=kmeans.labels_
print(labels)
color=['g.','r.','c.','y.']
for lin in range(len(X)):
    plt.plot(X[lin][0],X[lin][1],color[labels[lin]],markersize=10)
    plt.scatter(centroid[:,0],centroid[:,1],marker='x',s=150,linewidths=5,zorder=10)
    plt.xlabel('X')
    plt.ylabel('Y')
plt.show()


