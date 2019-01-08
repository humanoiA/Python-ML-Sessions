import pandas as pa
import numpy as np
from pylab import *
df=pa.DataFrame({'name':['HEMA','JAYA','REKHA','SUSHMA','RICHA','SHUBHANG'],'State':['UP','MP','HP','DEL','OR','BR']
                 ,'Grade':['A','B','C','E','F','A'],'Age':np.random.uniform(5000,30000,size=6)})
df.boxplot()#gives boxplot of single variable
show()