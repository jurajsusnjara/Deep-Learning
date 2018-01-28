import tensorflow as tf
import utils
import numpy as np
import matplotlib.pyplot as plt


Nh = 100
Nv = 784
h1_shape = (10, 10)
v_shape = (28, 28)
Nu = 5000
Nh2 = Nh  # Broj elemenata drugog skrivenog sloja
h2_shape = h1_shape

batch_size = 100
epochs = 100
n_samples = utils.mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

gibbs_sampling_steps = 2
alpha = 0.1
beta = 0.01

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

    v1_prob = tf.sigmoid(tf.add(tf.matmul(h1, tf.transpose(update_w1)), update_vb1))
    v1 = utils.sample_prob(v1_prob)

    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.global_variables_initializer()

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)
print('Training first RBM')
for i in range(total_batch):
    batch, _ = utils.mnist.train.next_batch(batch_size)
    err, out = sess1.run([err_sum1, out1], feed_dict={X1: batch})
    if i % (int(total_batch / 10)) == 0:
        print(i, err)
vr1 = sess1.run(v1_prob, feed_dict={X1: utils.teX[0:Nu, :]})
w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)

g2 = tf.Graph()
with g2.as_default():
    X2 = tf.placeholder("float", [None, Nv])
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = utils.weights([Nh, Nh2])
    # w2 = tf.Variable(tf.transpose(w1a))
    hb2 = utils.bias([Nh2])

    v0 = utils.sample_prob(X2)
    h1up_prob = tf.sigmoid(tf.add(tf.matmul(v0, w1a), hb1a))
    h1up = utils.sample_prob(h1up_prob)
    h2up_prob = tf.sigmoid(tf.add(tf.matmul(h1up, w2), hb2))
    h2up = utils.sample_prob(h2up_prob)
    h2down = h2up

    for step in range(gibbs_sampling_steps):
        h1down_prob = tf.sigmoid(tf.add(tf.matmul(h2down, tf.transpose(w2)), hb1a))
        h1down = utils.sample_prob(h1down_prob)
        h2down_prob = tf.sigmoid(tf.add(tf.matmul(h1down, w2), hb2))
        h2down = utils.sample_prob(h2down_prob)

    w2_positive_grad = tf.matmul(tf.transpose(h1up), h2up)
    w2_negative_grad = tf.matmul(tf.transpose(h1down), h2down)

    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1up)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_hb1a = tf.assign_add(hb1a, alpha * tf.reduce_mean(h1up - h1down, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2up - h2down, 0))

    out2 = (update_w2, update_hb1a, update_hb2)

    # rekonstrukcija ulaza na temelju krovnog skrivenog stanja h3
    # ...
    # ...
    h1_prob = tf.sigmoid(tf.add(tf.matmul(h2down, tf.transpose(update_w2)), update_hb1a))
    h1 = utils.sample_prob(h1_prob)
    v_out_prob = tf.sigmoid(tf.add(tf.matmul(h1, tf.transpose(w1a)), vb1a))
    v_out = utils.sample_prob(v_out_prob)

    err2 = X2 - v_out_prob
    err_sum2 = tf.reduce_mean(err2 * err2)

    initialize2 = tf.global_variables_initializer()

sess2 = tf.Session(graph=g2)
sess2.run(initialize2)
print('Training second RBM')
for i in range(total_batch):
    # iteracije treniranja
    batch, _ = utils.mnist.train.next_batch(batch_size)
    err, _ = sess2.run([err_sum2, out2], feed_dict={X2: batch})
    if i % (int(total_batch / 10)) == 0:
        print(i, err)
    w2s, hb1as, hb2s = sess2.run([w2, hb1a, hb2], feed_dict={X2: batch})
    vr2, h2downs = sess2.run([v_out_prob, h2down], feed_dict={X2: utils.teX[0:Nu, :]})

g3 = tf.Graph()
with g3.as_default():
    X3 = tf.placeholder("float", [None, Nv])
    r1_up = tf.Variable(w1s)
    w1_down = tf.Variable(tf.transpose(w1s))
    w2a = tf.Variable(w2s)
    hb1_up = tf.Variable(hb1s)
    hb1_down = tf.Variable(hb1as)
    vb1_down = tf.Variable(vb1s)
    hb2a = tf.Variable(hb2s)

    # wake pass
    v0 = utils.sample_prob(X3)
    h1_up_prob = tf.sigmoid(tf.add(tf.matmul(v0, r1_up), hb1_up))
    h1_up = utils.sample_prob(h1_up_prob) # s^{(n)} u pripremi
    v1_up_down_prob = tf.sigmoid(tf.add(tf.matmul(h1_up, w1_down), vb1_down))
    v1_up_down = utils.sample_prob(v1_up_down_prob) # s^{(n-1)\mathit{novo}} u tekstu pripreme

    # top RBM Gibs passes
    h2_up_prob = tf.sigmoid(tf.add(tf.matmul(h1_up, w2a), hb2a))
    h2_up = utils.sample_prob(h2_up_prob)
    h2_down = h2_up
    for step in range(gibbs_sampling_steps):
        h1_down_prob = tf.sigmoid(tf.add(tf.matmul(h2_down, tf.transpose(w2a)), hb1_down))
        h1_down = utils.sample_prob(h1_down_prob)
        h2_down_prob = tf.sigmoid(tf.add(tf.matmul(h1_down, w2a), hb2a))
        h2_down = utils.sample_prob(h2_down_prob)

    # sleep pass
    v1_down_prob = tf.sigmoid(tf.add(tf.matmul(h1_down, w1_down), vb1_down))
    v1_down = utils.sample_prob(v1_down_prob) # s^{(n-1)} u pripremi
    h1_down_up_prob = tf.sigmoid(tf.add(tf.matmul(v1_down, r1_up), hb1_up))
    h1_down_up = utils.sample_prob(h1_down_up_prob) # s^{(n)\mathit{novo}} u pripremi

    # generative weights update during wake pass
    update_w1_down = tf.assign_add(w1_down, beta * tf.matmul(tf.transpose(h1_up), X3 - v1_up_down_prob) / tf.to_float(
        tf.shape(X3)[0]))
    update_vb1_down = tf.assign_add(vb1_down, beta * tf.reduce_mean(X3 - v1_up_down_prob, 0))

    # top RBM update
    w2_positive_grad = tf.matmul(tf.transpose(h1_up), h2_up)
    w2_negative_grad = tf.matmul(tf.transpose(h1_down), h2_down)
    dw3 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1_up)[0])
    update_w2 = tf.assign_add(w2a, beta * dw3)
    update_hb1_down = tf.assign_add(hb1_down, beta * tf.reduce_mean(h1_up - h1_down, 0))
    update_hb2 = tf.assign_add(hb2a, beta * tf.reduce_mean(h2_up - h2_down, 0))

    # recognition weights update during sleep pass
    update_r1_up = tf.assign_add(r1_up,
                                 beta * tf.matmul(tf.transpose(v1_down_prob), h1_down - h1_down_up) / tf.to_float(
                                     tf.shape(X3)[0]))
    update_hb1_up = tf.assign_add(hb1_up, beta * tf.reduce_mean(h1_down - h1_down_up, 0))

    out3 = (update_w1_down, update_vb1_down, update_w2, update_hb1_down, update_hb2, update_r1_up, update_hb1_up)

    err3 = X3 - v1_down_prob
    err_sum3 = tf.reduce_mean(err3 * err3)

    initialize3 = tf.global_variables_initializer()

sess3 = tf.Session(graph=g3)
sess3.run(initialize3)
print('Fine tuning DBN')
for i in range(total_batch):
    batch, _ = utils.mnist.train.next_batch(batch_size)
    err, _ = sess3.run([err_sum3, out3], feed_dict={X3: batch})
    if i % (int(total_batch / 10)) == 0:
        print(i, err)
    w2ss, r1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs = sess3.run(
        [w2a, r1_up, w1_down, hb2a, hb1_up, hb1_down, vb1_down], feed_dict={X3: batch})
    vr3, h2_downs, h2_down_probs = sess3.run([v1_down_prob, h2_down, h2_down_prob], feed_dict={X3: utils.teX[0:Nu, :]})

# vizualizacija težina
utils.draw_weights(r1_ups, v_shape, Nh, h1_shape, name='weights_r1_ups.png')
utils.draw_weights(w1_downs.T, v_shape, Nh, h1_shape, name='weights_w1_downs.png')
utils.draw_weights(w2ss, h1_shape, Nh2, h2_shape, interpolation="nearest", name='weights_w2ss.png')

Npics = 5
plt.figure(figsize=(8, 12 * 4))
for i in range(20):
    plt.subplot(20, Npics, Npics * i + 1)
    plt.imshow(utils.teX[i].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Test input")
    plt.subplot(20, Npics, Npics * i + 2)
    plt.imshow(vr1[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 1")
    plt.subplot(20, Npics, Npics * i + 3)
    plt.imshow(vr2[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 2")
    plt.subplot(20, Npics, Npics * i + 4)
    plt.imshow(vr3[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 3")
    plt.subplot(20, Npics, Npics * i + 5)
    plt.imshow(h2_downs[i][0:Nh2].reshape(h2_shape), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Top states 3")
plt.tight_layout()
plt.savefig('figures/reconstructions.png')



# # vizualizacija težina
# utils.draw_weights(w2s, h1_shape, Nh2, h2_shape, interpolation="nearest")
#
# # vizualizacija rekonstrukcije i stanja
# utils.draw_reconstructions(utils.teX, vr2, h2downs, v_shape, h2_shape, 20)
#
# Generiranje uzoraka iz slučajnih vektora krovnog skrivenog sloja
r_input = np.random.rand(100, Nh2)
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


out_h1_prob = utils.sigmoid(np.dot(r_input, w2ss.T) + hb1_downs)
out_h1 = np.random.rand(out_h1_prob.shape[0], out_h1_prob.shape[1])
out_v_prob = utils.sigmoid(np.dot(out_h1, w1_downs) + vb1_downs)
out_1 = np.random.rand(out_v_prob.shape[0], out_v_prob.shape[1]) <= out_v_prob

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict

for i in range(1000):
    out_1_prob, out_1, hout1 = sess3.run((v1_down_prob, v1_down, h2_down), feed_dict={X3: out_1})

utils.draw_generated(r_input, hout1, out_1_prob, v_shape, h2_shape, 50)
