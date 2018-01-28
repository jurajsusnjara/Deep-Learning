import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch, label = mnist.train.next_batch(20)


X1 = tf.placeholder("float", [None, 784])
inp_plc = tf.placeholder("float", [784, 100])
w1 = weights([784, 100])
vb1 = bias([100])
S = tf.sigmoid(tf.add(tf.matmul(X1, w1), vb1))

upd_w1 = tf.add(w1, inp_plc)
R = tf.sigmoid(tf.add(tf.matmul(X1, upd_w1), vb1))


sesh = tf.Session()
sesh.run(tf.global_variables_initializer())


inp = np.ones((784, 100))
res1 = sesh.run([S, R], feed_dict={X1: batch, inp_plc: inp})
res2 = sesh.run([S, R], feed_dict={X1: batch, inp_plc: inp})
