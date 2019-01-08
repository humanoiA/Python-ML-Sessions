import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
x=[1,2,3,4,5,6,7,8]
y=[5,3,9,2,9,1,4,6]
pt.plot(x,y)
pt.show()
pt.scatter(x,y,color="magenta")
pt.xlabel("X-AXIS")
pt.ylabel("Y-AXIS")
pt.title("TEST GRAPH")
pt.xlim(-5,20)
pt.ylim(-5,15)
pt.show()