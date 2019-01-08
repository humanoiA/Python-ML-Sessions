import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
x,y=mglearn.datasets.make_wave(n_samples=60)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
lr=LinearRegression().fit(x_train,y_train)
print('lr coef: %s' %lr.coef_)
print('lr intercept: %s' %lr.intercept_)
print('training set score: %s'%lr.score(x_train,y_train))
print('test set score : %s'%lr.score(x_test,y_test))
x,y=mglearn.datasets.load_extended_boston()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
lr=LinearRegression().fit(x_train,y_train)
print('training set score: %s'%lr.score(x_train,y_train))
print('test set score : %s'%lr.score(x_test,y_test))
from sklearn.linear_model import Ridge
ridge=Ridge().fit(x_train,y_train)
print('training set score: %s'%ridge.score(x_train,y_train))
print('test set score : %s'%ridge.score(x_test,y_test))
ridge10=Ridge(alpha=10).fit(x_train,y_train)
print('training set score: %s'%ridge10.score(x_train,y_train))
print('test set score : %s'%ridge10.score(x_test,y_test))
ridge01=Ridge(alpha=0.01).fit(x_train,y_train)
print('training set score: %s'%ridge01.score(x_train,y_train))
print('test set score : %s'%ridge01.score(x_test,y_test))
plt.title('Ridge Coefficient')
plt.plot(ridge.coef_,'o',label='Alpha')
plt.plot(ridge10.coef_,'o',label='Alpha10')
plt.plot(ridge01.coef_,'o',label='Alpha0.01')
plt.plot(lr.coef_,'o',label='Linear')
plt.ylim(-25,25)
plt.legend()
plt.show()
from sklearn.linear_model import Lasso
lasso=Lasso().fit(x_train,y_train)
print('training set score: %s'%lasso.score(x_train,y_train))
print('test set score : %s'%lasso.score(x_test,y_test))
print('Number of Features used %d'%np.sum(lasso.coef_!=0))
lasso001=Lasso(alpha=0.01).fit(x_train,y_train)
print('training set score: %s'%lasso001.score(x_train,y_train))
print('test set score : %s'%lasso001.score(x_test,y_test))
print('Number of Features used %d'%np.sum(lasso001.coef_!=0))
lasso0001=Lasso(alpha=0.0001).fit(x_train,y_train)
print('training set score: %s'%lasso0001.score(x_train,y_train))
print('test set score : %s'%lasso0001.score(x_test,y_test))
print('Number of Features used %d'%np.sum(lasso0001.coef_!=0))
plt.title('Lasso Coefficient')
plt.plot(lasso.coef_,'o',label='Alpha')
plt.plot(lasso001.coef_,'o',label='Alpha0.01')
plt.plot(lasso0001.coef_,'o',label='Alpha0.001')
#plt.plot(lr.coef_,'o',label='Linear')
plt.ylim(-25,25)
plt.legend()
plt.show()