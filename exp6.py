import  pandas as pa
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,20,1000)
y=np.sin(x)
plt.plot(x,y,label="My First Application")
plt.title("My simple data display")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid(True)
plt.figtext(0.995,0.01,'Footnote',ha='right',va='bottom')
plt.legend(loc='best',framealpha=0.5,prop={'size':'small'})
plt.tight_layout(pad=1)
plt.savefig('test.png')
plt.show()