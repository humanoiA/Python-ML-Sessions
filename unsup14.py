import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster import KMeans
dataset=pd.read_excel('titanic.xls')
dataset.drop(['name','ticket','cabin','embarked','boat','home.dest'],axis=1,inplace=True)
print(dataset)
dataset=dataset.fillna(0).reset_index()
l=preprocessing.LabelEncoder()
dataset['sex']=l.fit_transform(dataset['sex'])
dataset['age']=l.fit_transform(dataset['age'])
dataset['sibsp']=l.fit_transform(dataset['sibsp'])
dataset['parch']=l.fit_transform(dataset['parch'])
#dataset['ticket']=l.fit_transform(dataset['ticket'])
dataset['fare']=l.fit_transform(dataset['fare'])
#dataset['cabin']=l.fit_transform(dataset['cabin'])
#dataset['embarked']=l.fit_transform(dataset['embarked'])
#dataset['boat']=l.fit_transform(dataset['boat'])
dataset['body']=l.fit_transform(dataset['body'])
normalizer=preprocessing.Normalizer()
n=normalizer.fit_transform(dataset)
print(n)
wcas=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=50)
    kmeans.fit(dataset)
    wcas.append(kmeans.inertia_)
plt.plot(range(1,11),wcas)
plt.title('My Elbow Demo')
plt.xlabel('Number of cluster')
plt.ylabel('WCAS')
plt.show()
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(dataset)
r=pd.DataFrame(y_kmeans)
print(y_kmeans)

writer = pd.ExcelWriter('MallData.xlsx')
r.to_excel(writer,'Sheet1',index=False)
writer.save()