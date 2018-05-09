import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]))+0.1
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        out_puts = Wx_plus_b
    else:
        out_puts = activation_function(Wx_plus_b)
    return out_puts

data = pd.read_excel('C:/Users/Administrator/Desktop/data.xlsx')
x_data = np.array(data)[:,1].reshape(155,1)
y_data = np.array(data)[:,0].reshape(155,1)

# x_data = []
# y_data = []
# x_ = np.array(data)[:,1]
# y_ = np.array(data)[:,0]
# for i in x_:
#     x_data.append(i)
# for i in y_:
#     y_data.append(i)
# x_data = np.array(x_data)
# y_data = np.array(y_data)

# x_data = np.linspace(-1,1,300,dtype=tf.float32)[:,np.newaxis]
# y_data = np.square(x_data)

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.sigmoid)
predition = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition,feed_dict={xs:x_data,ys:y_data})
        lines = ax.plot(x_data,predition_value,'r-',lw=1)
        plt.pause(0.1)







