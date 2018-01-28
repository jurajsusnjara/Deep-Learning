import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_deep import TFDeep
import data
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split


tf.app.flags.DEFINE_string('data_dir',
  '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(
  tf.app.flags.FLAGS.data_dir, one_hot=True)

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]
print(N, D, C)

X_train, X_val, y_train, y_val = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.8)
