import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
fig,ax=plt.subplots()
fig,(ax1,ax2,ax3)=plt.subplots(nrows=3,ncols=1,sharex=False)
figsize=(8,4)
for ax in fig.get_axes():
    pass
plt.show()