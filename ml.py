import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

# model parameters
w=tf.Variable([0.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)
w_matrix=[]
b_matrix=[]
loss_matrix=[]

#input and output parameters
x=tf.placeholder(dtype=tf.float32)
linear_model=w*x+b
y=tf.placeholder(dtype=tf.float32)

#loss
loss=tf.reduce_sum(tf.square(linear_model-y))

#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

#trainning
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]
# x_train=[5,8,1,12.35]
# y_train=[0.01317,0.02107,0.002634,0.03253]
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(5000):
    c_w,c_b,c_loss,c_train = sess.run([w,b,loss,train],{x:x_train,y:y_train})
    w_matrix.append(c_w[0])
    b_matrix.append(c_b[0])
    loss_matrix.append(c_loss)

print (sess.run(loss,{x:x_train,y:y_train}))
print (sess.run(linear_model,{x:x_train}))
# print (loss_matrix)
# print (w_matrix)
# print (b_matrix)
# # print (w_matrix,b_matrix)
# # curr_w,curr_b,loss=sess.run([w,b,loss],{x:x_train,y:y_train})
# # print ("w:%s b:%s loss:%s"%(curr_w,curr_b,loss))
#
#
# plt.figure("gradient descent")
# plt.plot(w_matrix,loss_matrix,"r.")
# plt.show()