import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
#center=[[1,1],[5,5],[3,10]]
x,y=make_blobs(n_samples=200,cluster_std=1)
plt.scatter(x[:,0],x[:,1])
plt.show()
ms=MeanShift()
ms.fit(x)
labels=ms.labels_
cluster_center=ms.cluster_centers_
color=['r.','g.','b.','c.','k.','y.','m.']
for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], color[labels[i]], markersize=10)
    plt.scatter(cluster_center[:, 0], cluster_center[:, 1], marker='x', zorder=10, s=150, linewidths=5)
plt.show()