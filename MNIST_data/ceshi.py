import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_data/",one_hot=True)

# print (mnist.train.next_batch(1))

x=tf.placeholder(tf.float32,[None,784])
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))


y=tf.nn.softmax(tf.matmul(x,w)+b)

y_=tf.placeholder(tf.float32,[None,10])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for xx in range(1000):
    bx,by=mnist.train.next_batch(500)
    # bx=mnist.train.images
    # by=mnist.train.labels
    sess.run(train_step,feed_dict={x:bx,y_:by})

correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print (sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels}))