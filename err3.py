from numpy import *
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
ex=[0.0,0.5,0.0,0.5,0.0]
pr=[0.2,0.3,0.4,0.1,0.6]
#mean forcast-average of forcast value
#mean absolute error=average of forcast values where all the values are positive
forcast_error=[ex[i]-pr[i]for i in range(len(ex))]
#mae=mean(abs(forcast_error))
#mse=mean(sqr(forcast_error))
print(mean_absolute_error(ex,pr))
print(mean_squared_error(ex,pr))
print(sqrt(mean_squared_error(ex,pr)))
