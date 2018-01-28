import tensorflow as tf
import utils


beta = 0.01

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
    v1_up_down_prob = tf.sigmoid(tf.add(tf.matmul(h1_up, tf.transpose(w1_down)), vb1_down))
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
    v1_down_prob = tf.sigmoid(tf.add(tf.matmul(h1_down, tf.transpose(w1_down)), vb1_down))
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

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess3 = tf.Session(graph=g3)
sess3.run(initialize3)
for i in range(total_batch):
    # ...
    err, _ = sess3.run([err_sum3, out3], feed_dict={X3: batch})

    if i % (int(total_batch / 10)) == 0:
        print(i, err)

    w2ss, r1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs = sess3.run(
        [w2a, r1_up, w1_down, hb2a, hb1_up, hb1_down, vb1_down], feed_dict={X3: batch})
    vr3, h2_downs, h2_down_probs = sess3.run([v1_down_prob, h2_down, h2_down_prob], feed_dict={X3: teX[0:Nu, :]})

# vizualizacija težina
draw_weights(r1_ups, v_shape, Nh, h1_shape)
draw_weights(w1_downs.T, v_shape, Nh, h1_shape)
draw_weights(w2ss, h1_shape, Nh2, h2_shape, interpolation="nearest")

# vizualizacija rekonstrukcije i stanja
Npics = 5
plt.figure(figsize=(8, 12 * 4))
for i in range(20):
    plt.subplot(20, Npics, Npics * i + 1)
    plt.imshow(teX[i].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Test input")
    plt.subplot(20, Npics, Npics * i + 2)
    plt.imshow(vr[i][0:784].reshape(v_shape), vmin=0, vmax=1)
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