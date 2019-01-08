import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel('data.xlsx')
data = df.values
x1=data[:,2:9]
print(x1)
#df1=df.drop(['ID Tag','Model','Department'])
#print(df1)
km=KMeans(n_clusters=7,init='k-means++',n_init=10)
km.fit(x1)
x=km.fit_predict(x1)
df['Cluster']=x
print(df.groupby(['Cluster']).mean())
sns.lmplot('WO count','Labor Cost',data=df,fit_reg=False,hue='Cluster',scatter_kws={'marker':'X','s':20})
plt.title('Low Work Order and Labor Cost')
plt.xlabel('Low Work Order')
plt.ylabel('Labor Count')
plt.show()
