from sklearn import linear_model
import matplotlib.pyplot as plt
feature = [[100],[200],[300],[400],[500]]
labels=[500,1000,1500,2000,2500]
plt.scatter(feature,labels,color='green')
plt.xlabel('Monthly Water Units')
plt.ylabel('Price')
clf=linear_model.LinearRegression()
clf=clf.fit(feature,labels)
result=clf.predict([[250],[420]])
plt.plot([[250],[420]],result,color='blue')
print('Result',result)
plt.show()