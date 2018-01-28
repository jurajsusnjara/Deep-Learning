import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## 1. definicija računskog grafa
# podatci i parametri
X  = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model
Y = a*X + b

# kvadratni gubitak
loss = tf.reduce_sum((Y-Y_)**2)
dL_dY = 2*(Y-Y_)
dL_da = tf.reduce_sum(X*dL_dY)
dL_db = tf.reduce_sum(dL_dY)

# optimizacijski postupak: gradijentni spust
trainer = tf.train.GradientDescentOptimizer(1e-4)

# train_op = trainer.minimize(loss)
grad_var_pairs = trainer.compute_gradients(loss, [a, b])
grads = [g[0] for g in grad_var_pairs]
train_op = trainer.apply_gradients(grad_var_pairs)

## 2. inicijalizacija parametara
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## 3. učenje
# neka igre počnu!
xx = [i for i in range(100)]
yy = [2*xx[i]+5 for i in range(100)]
for i in range(100):
    var_list = [loss, train_op, a, b, grads[0], grads[1], dL_da, dL_db]
    val_loss, _, val_a, val_b, grad_a, grad_b, ga, gb = sess.run(var_list, feed_dict={X: xx, Y_: yy})
    print(i, val_loss, val_a, val_b, grad_a, grad_b, ga, gb)
