import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
import utils

Nh = 100  # Broj elemenata prvog skrivenog sloja
h1_shape = (10, 10)
Nv = 784  # Broj elemenata vidljivog sloja
v_shape = (28, 28)
Nu = 5000  # Broj uzoraka za vizualizaciju rekonstrukcije

gibbs_sampling_steps = 1
alpha = 0.1

g1 = tf.Graph()
with g1.as_default():
    X1 = tf.placeholder("float", [None, 784])
    w1 = utils.weights([Nv, Nh])
    vb1 = utils.bias([Nv])
    hb1 = utils.bias([Nh])

    v0 = utils.sample_prob(X1)
    # v0 = X1
    h0_prob = tf.sigmoid(tf.add(tf.matmul(v0, w1), hb1))
    h0 = utils.sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = tf.sigmoid(tf.add(tf.matmul(h1, tf.transpose(w1)), vb1))
        v1 = utils.sample_prob(v1_prob)
        h1_prob = tf.sigmoid(tf.add(tf.matmul(v1, w1), hb1))
        h1 = utils.sample_prob(h1_prob)

    w1_positive_grad = tf.matmul(tf.transpose(v0), h0)
    w1_negative_grad = tf.matmul(tf.transpose(v1), h1)

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0))

    out1 = (update_w1, update_vb1, update_hb1)

    # h_p = tf.sigmoid(tf.add(tf.matmul(v0, update_w1), update_hb1))
    # h_ = sample_prob(h_p)
    v1_prob = tf.sigmoid(tf.add(tf.matmul(h1, tf.transpose(update_w1)), update_vb1))
    v1 = utils.sample_prob(v1_prob)

    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 1 # default 100
n_samples = utils.mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = utils.mnist.train.next_batch(batch_size)
    err, out = sess1.run([err_sum1, out1], feed_dict={X1: batch})

    if i % (int(total_batch / 10)) == 0:
        print(i, err)

w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)
vr, h1s = sess1.run([v1_prob, h1], feed_dict={X1: utils.teX[0:Nu, :]})

# vizualizacija težina
utils.draw_weights(w1s, v_shape, Nh, h1_shape)

# vizualizacija rekonstrukcije i stanja
utils.draw_reconstructions(utils.teX, vr, h1s, v_shape, h1_shape, 200)

utils.reconstruct(0, h1s, utils.teX, w1s, vb1s, h1_shape, v_shape, Nh)  # prvi argument je indeks znamenke u matrici znamenki

# Vjerojatnost da je skriveno stanje uključeno kroz Nu ulaznih uzoraka
plt.figure()
tmp = (h1s.sum(0) / h1s.shape[0]).reshape(h1_shape)
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('vjerojatnosti (ucestalosti) aktivacije pojedinih neurona skrivenog sloja')
plt.savefig('figures/probabilities_of_activations.png')

# Vizualizacija težina sortitranih prema učestalosti
tmp_ind = (-tmp).argsort(None)
utils.draw_weights(w1s[:, tmp_ind], v_shape, Nh, h1_shape)
plt.title('Sortirane matrice tezina - od najucestalijih do najmanje koristenih')


# Generiranje uzoraka iz slučajnih vektora
r_input = np.random.rand(100, Nh)
r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20  # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s

# (update_w1, update_vb1, update_hb1)
v_prob = utils.sigmoid(np.dot(r_input, w1s.T) + vb1s)
out_1 = np.random.rand(v_prob.shape[0], v_prob.shape[1]) <= v_prob

# out_1 = sess1.run((v1), feed_dict={h0: r_input})

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
for i in range(1000):
    out_1_prob, out_1, hout1 = sess1.run((v1_prob, v1, h1), feed_dict={X1: out_1})

utils.draw_generated(r_input, hout1, out_1_prob, v_shape, h1_shape, 50)
