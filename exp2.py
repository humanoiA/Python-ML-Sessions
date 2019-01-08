from pandas import *
from pylab import *
import numpy as np
data1=read_csv('healthcare_Sample_dataset11.csv',converters={'AGE':int})
lp=data1['ID']
name=data1['NAME']
dob=data1['DOB']
mob=data1['MOB']
mail=data1['MAIL']
ssn=data1['SSN']
gen=data1['GEN']
disease=data1['DISEASE']
age=data1['AGE']


mydata=list(zip(lp,name,dob,mob,mail,ssn,gen,disease,age))
df= DataFrame(data=mydata,columns=['ID','NAME','DOB','MOB','MAIL','SSN','GEN','DISEASE','AGE'])
sorteddata=df.sort_values(['AGE'],ascending=False)
#print(sorteddata)
print(df['DISEASE'].unique())
#df.groupby('DISEASE')['AGE'].mean().plot(kind='bar')
#show()
#df.groupby('GEN')['DISEASE'].count().plot(kind='bar')
#show()
df.groupby('GEN')('DISEASE').count().plot(kind='bar')
#show()
