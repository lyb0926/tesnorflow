import math
import matplotlib.pyplot as plt
import numpy as np
x2= []
y1= []
y2= []
y3= []
x1 = np.linspace(0,5,100)
for x in x1:
    x2.append(x)
    y_out1 = math.cos(x)+math.sin(x)
    y1.append(y_out1)
    y_out2 = 2* math.cos(x) + 2* math.sin(x)
    y2.append(y_out2)
    y_out3 = 3 * math.cos(x) + 3 * math.sin(x)
    y3.append(y_out3)
plt.plot(x2,y1,'r-')
plt.plot(x2,y2,'b-')
plt.plot(x2,y3,'g-')
plt.show()

