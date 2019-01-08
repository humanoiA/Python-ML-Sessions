from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from plotly.offline import init_notebook_mode
import pandas as pd
#init_notebook_mode()
p1=pd.read_excel('RESPONSEDATA.xlsx','Campaign')
p2=pd.read_excel('RESPONSEDATA.xlsx','Response')
p2['n']=1
print(p1.tail())
print(p2.tail())
df=pd.merge(p1,p2,on='CampaignID')
print(df.tail())
table=df.pivot_table(index=['Patient'],columns=['CampaignID'],values='n')
print(table.tail())
table=table.fillna(0).reset_index()
cols=table.columns[1:]
print(cols)
cluster=KMeans(n_clusters=5)
table['cluster']=cluster.fit_predict(table[table.columns[2:]])
pca=PCA(n_components=2)
table['x']=pca.fit_transform(table[cols])[:,0]
table['y']=pca.fit_transform(table[cols])[:,1]
table=table.reset_index()
print(table.tail())
p_cluster=table[['Patient','cluster','x','y']]
print(p_cluster.tail())
f=pd.merge(p2,p_cluster)
f=pd.merge(p1,f)
print(f.tail())
f['0']=f.cluster==0
print(f.groupby('0').Type.value_counts())