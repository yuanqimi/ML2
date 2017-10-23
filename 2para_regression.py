import tensorflow as tf
import numpy as np
import os
import tensorflow

w=tf.Variable(tf.zeros([2,1]),dtype=tf.float32)
b=tf.Variable(tf.zeros([1]),dtype=tf.float32)
x=tf.placeholder(tf.float32,[None,2])
y=tf.placeholder(tf.float32,[None,1])

linear_mod=tf.matmul(x,w)+b

loss=tf.reduce_sum(tf.square(linear_mod-y))

train=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
x_train=sess.run(tf.transpose([[0.020,0.020,0.020,0.020,0.020,0.020,0.021,0.021,0.021,0.021,0.021,0.021,0.022,0.022,0.022,0.022,0.022,0.022],[0.07,0.08,0.09,0.010,0.011,0.012,0.07,0.08,0.09,0.010,0.011,0.012,0.07,0.08,0.09,0.010,0.011,0.012]]))
y_train=sess.run(tf.transpose([[3.44922,3.93573,4.43336,4.9217,5.41378,5.91141,3.4318,3.91589,4.40367,4.90101,5.38426,5.87293,3.41158,3.89434,4.38667,4.86678,5.35568,5.84021]]))

for i in range(800000):
    sess.run(train,feed_dict={x:x_train,y:y_train})


print (sess.run(loss,{x:x_train,y:y_train}))



print (sess.run(linear_mod,{x:sess.run(tf.transpose([[0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203,0.0203],[0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085,0.0085]]))}))