from sklearn.datasets import make_blobs,load_iris
import mglearn
import matplotlib.pyplot as plt
import numpy as np
x,y=make_blobs(random_state=42)
#plt.scatter(x[:,0],x[:,1],c=y,s=60,cmap=mglearn.cm3)
#plt.show()
#x,y=load_iris(True)
#plt.scatter(x[:,0],x[:,1],c=y,s=60,cmap=mglearn.cm3)
#plt.show()
from sklearn.svm import LinearSVC
linear_svc=LinearSVC().fit(x,y)
print(linear_svc.coef_.shape)
print(linear_svc.intercept_.shape)
plt.scatter(x[:,0],x[:,1],c=y,s=60,cmap=mglearn.cm3)
line=np.linspace(-15,15)
for coef,intercept in zip(linear_svc.coef_,linear_svc.intercept_):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1])
    plt.ylim(-10,15)
    plt.xlim(-10,8)
plt.show()
mglearn.plots.plot_2d_classification(linear_svc,x,fill=True,alpha=0.7)
plt.scatter(x[:,0],x[:,1],c=y,s=60)
line=np.linspace(-15,15)
for coef,intercept in zip(linear_svc.coef_,linear_svc.intercept_):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1])
plt.show()