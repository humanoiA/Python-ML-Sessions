import pandas as pd
from pylab import *
from numpy import random
names=['Ashish','Amit','Dinesh','Priya','Shubhang']
dob=[1984,1986,1999,1994,2000]
mydataset=list(zip(names,dob))
#print(mydataset)
df=pd.DataFrame(data=mydataset,columns=['Name','DOB'])#columns in data frame
print(df)
#loc=r'bdaydata.csv'
#df.to_csv('bdaydata.csv',index=False,header=False)
#file="bdaydata.csv"
#df=pd.read_csv(file,name=['Name','DOB'])#name in csv or excel
#print(df)
#import os
#os.remove(file)
print(df.dtypes)
sorteddata=df.sort_values(['DOB'],ascending=False)
print(sorteddata)
print(sorteddata.head(1))
print(df['DOB'].max())
print(df['DOB'].min())
maxn=df['Name'][df['DOB']==df['DOB'].max()].values
print(maxn)
print(df[df['DOB']==df['DOB'].max()])
random.seed(500)#to generate same data again and again
rand_names=[names[random.randint(low=0,high=len(names))]    for i in range(1000)]
print(rand_names[:10])
rand_dob=[dob[random.randint(low=0,high=len(dob))]  for i in range(1000)]
print(rand_dob[:10])
mynewdata=list(zip(rand_names,rand_dob))
print(mynewdata[:10])
ca=pd.DataFrame(data=mynewdata,columns=['Name','DOB'])
print(ca[:10])
print(ca['Name'].unique())
print(ca['Name'].describe())
name=ca.groupby('Name')
print(name)
ca['DOB'].plot.bar()
show()
print(ca.sort_values(by='DOB',ascending=False))
df['AVG DOB']=df['DOB']*100/len(df)
print(df[:10])
a=df[df.Name=='Shubhang']
print(a)