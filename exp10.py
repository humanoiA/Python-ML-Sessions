import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
data1=pd.read_csv("cgpa.csv")
df=data1.head(50)
pt.plot(df['rollno'],df['cgpa'],color='orange',label='line')
pt.scatter(df['rollno'],df['cgpa'],color='blue',label='scatter')
pt.legend()
pt.xlabel('ROLL')
pt.ylabel('CGPA')
pt.title("CGPA vs ROLL")
pt.show()
pt.barh(df['rollno'],df['cgpa'],color=['orange','blue'],label='bar',alpha=.7)
pt.legend()
pt.xlabel('ROLL')
pt.ylabel('CGPA')
pt.title("CGPA vs ROLL")
pt.show()
