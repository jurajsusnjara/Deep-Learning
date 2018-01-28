import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [2, 2])
Y = 3 * X + 5
z = Y[0, 0]
sess = tf.Session()
Y_val = sess.run(Y, feed_dict={X: [[0,1],[2,3]]})
z_val = sess.run(z, feed_dict={X: np.ones(2,2)})

print(Y_val[0,0], type(Y_val))
print(z_val, type(z_val))
