import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x,y):
    #no. of elements
    n=np.size(x)
    x_mean,y_mean=np.mean(x),np.mean(y)
    #cross validation & deviation about x
    ss_xy=np.sum(y*x-n*x_mean*y_mean)
    ss_xx=np.sum(x*x-n*x_mean*x_mean)
    b_1=ss_xy/ss_xx
    b_0=y_mean-b_1*x_mean
    return(b_0,b_1)
def plot_regression(x,y,b):
    plt.scatter(x,y,color='m',marker='o',s=30)
    y_pred=b[0]+b[1]*x
    plt.plot(x,y_pred,color='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
def main():
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    b=estimate_coef(x,y)
    print(b[0],b[1])
    plot_regression(x,y,b)
main()