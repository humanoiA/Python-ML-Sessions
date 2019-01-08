from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
data=make_blobs()
#x,y=data.data,data.target
x1,y1=make_blobs(n_samples=200,centers=3,cluster_std=1)
#print(x1)
plt.scatter(x1[:,0],x1[:,1],c='g',marker='x',edgecolors='black',s=50)
plt.grid()
plt.show()