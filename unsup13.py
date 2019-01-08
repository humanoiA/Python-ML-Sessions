import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
dataset=pd.read_csv('crime_data.csv')
x=dataset.iloc[:,[2,5]].values
wcas=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=50)
    kmeans.fit(x)
    wcas.append(kmeans.inertia_)
plt.plot(range(1,11),wcas)
plt.title('My Elbow Demo')
plt.xlabel('Number of cluster')
plt.ylabel('WCAS')
plt.show()
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(x)
r=pd.DataFrame(y_kmeans)
print(y_kmeans)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='CLS1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='CLS2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='CLS3')
#plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='black',label='CLS4')
#plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='magenta',label='CLS5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('Mall Analysis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
writer = pd.ExcelWriter('MallData.xlsx')
r.to_excel(writer,'Sheet1',index=False)
writer.save()