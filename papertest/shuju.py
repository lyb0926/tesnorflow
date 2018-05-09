import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x1_data = []
y1_data = []
x2_data = []
y2_data = []
data = pd.read_excel('C:/Users/Administrator/Desktop/data.xlsx')
for i in range(155):
    if np.array(data)[i,2] == 0:
        x1_data.append(np.array(data)[i,1])
        y1_data.append(np.array(data)[i,0])
for i in range(155):
    if np.array(data)[i,2] == 1:
        x2_data.append(np.array(data)[i,1])
        y2_data.append(np.array(data)[i,0])
plt.plot(x1_data,y1_data,"b.")
plt.plot(x2_data,y2_data,"r.")
plt.show()
