import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

w=tf.Variable([0.4],dtype=tf.float64)
b=tf.Variable([0.1],dtype=tf.float64)
x=tf.placeholder(dtype=tf.float64)
y=tf.placeholder(dtype=tf.float64)

linear_mod=w*x+b

loss=tf.reduce_sum(tf.square(linear_mod-y))

train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# x_train=[5,8,1,12.35]
# y_train=[0.01317,0.02107,0.002634,0.03253]
# x_eval=[0.5,23.456,9.999,15.986]
# y_eval=[0.001462,0.06178,0.02633,0.0421]

x_train=[0.020,0.021,0.022,0.023,0.024,0.025,0.026,0.030]
y_train=[4.9217,4.89002,4.85833,4.82665,4.79496,4.76328,4.7316,4.60486]
# x_eval=[0.027,0.028,0.029]
x_eval=[0.027,0.027,0.027]
y_eval=[4.69991,4.66823,4.63654]

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    cw, cb, c_train = sess.run([w,b,train],{x:x_train,y:np.ones(8)/y_train})
    print (cw)

# accu=0.1
# while accu > 1.0e-6:
#     cw, cb, c_train = sess.run([w,b,train],{x:x_train,y:np.ones(8)/y_train})
#     accu=sess.run(loss,{x:x_train,y:np.ones(8)/y_train})
#     print (cw)

# print (sess.run(loss,{x:x_eval,y:y_eval}))
print (sess.run(loss,{x:x_train,y:np.ones(8)/y_train}))
print(np.ones(3)/sess.run(linear_mod,{x:x_eval}))

# plt.figure("displacement prediction")
# plt.plot(x_train,y_train,"r*")
# plt.plot(x_train,np.ones(8)/sess.run(linear_mod,{x:x_train}),"b<")
# plt.show()