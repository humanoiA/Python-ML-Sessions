import mglearn as mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
x,y=mglearn.datasets.make_wave(n_samples=50)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(x_train,y_train)
KNeighborsRegressor(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=3,p=2,weights='uniform')
reg.predict(x_test)
reg.score(x_test,y_test)
fig,axes=plt.subplots(1,3,figsize=(15,4))
line=np.linspace(-3,3,1000).reshape(-1,1)
plt.suptitle('Neighbour Regression')
for n_neighbours,axis in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbours).fit(x,y)
    axis.plot(x,y,'o')
    axis.plot(x,-3*np.ones(len(x)),'o')
    axis.plot(line,reg.predict(line))
    axis.set_title('%d neighbours'%n_neighbours)
plt.show()