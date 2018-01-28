import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.layers as layers
import skimage as ski
import skimage.io
import os
import math


def test(test_x, test_y):
    print('Test error:')
    feed_dict = {node_x: test_x, node_y: test_y}
    valid_loss, valid_accuracy = sesh.run([loss, accuracy],
                                          feed_dict=feed_dict)
    print("Test loss: ", valid_loss)
    print("Test accuracy: ", valid_accuracy)


def draw_conv_filters(epoch, step, weights, save_dir):
      w = weights.copy()
      num_filters = w.shape[3]
      num_channels = w.shape[2]
      k = w.shape[0]
      assert w.shape[0] == w.shape[1]
      w = w.reshape(k, k, num_channels, num_filters)
      w -= w.min()
      w /= w.max()
      border = 1
      cols = 8
      rows = math.ceil(num_filters / cols)
      width = cols * k + (cols-1) * border
      height = rows * k + (rows-1) * border
      img = np.zeros([height, width])
      for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k] = w[:,:,0,i]
      filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
      ski.io.imsave(os.path.join(save_dir, filename), img)


def train(train_x, train_y, valid_x, valid_y):
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']

    trainer = tf.train.GradientDescentOptimizer(0.01)
    train_op = trainer.minimize(loss)
    sesh.run(tf.initialize_all_variables())

    for epoch_num in range(1, max_epochs + 1):
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        cnt_correct = 0

        for step in range(num_batches):
            offset = step * batch_size
            batch_x = train_x[offset:(offset + batch_size), ...]
            batch_y = train_y[offset:(offset + batch_size)]
            feed_dict = {node_x: batch_x, node_y: batch_y}
            start_time = time.time()
            run_ops = [train_op, loss, logits]
            ret_val = sesh.run(run_ops, feed_dict=feed_dict)
            _, loss_val, logits_val = ret_val
            yp = np.argmax(logits_val, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()

            duration = time.time() - start_time
            if (step + 1) % 50 == 0:
                sec_per_batch = float(duration)
                format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
                print(format_str % (epoch_num, step + 1, num_batches, loss_val, sec_per_batch))
                print("Train accuracy = %.2f" % (cnt_correct / ((step + 1) * batch_size) * 100))

            if step % 100 == 0:
                conv1_var = tf.contrib.framework.get_variables('conv1')[0]
                conv1_weights = conv1_var.eval(session=sesh)
                draw_conv_filters(epoch_num, step * batch_size, conv1_weights, save_dir)

        print('Train error:')
        feed_dict = {node_x: test_x, node_y: test_y}
        valid_loss, valid_accuracy = sesh.run([loss, accuracy],
                                              feed_dict=feed_dict)
        print("Train loss: ", valid_loss)
        print("Train accuracy: ", valid_accuracy)

        print('Validation error:')
        feed_dict = {node_x: valid_x, node_y: valid_y}
        valid_loss, valid_accuracy = sesh.run([loss, accuracy],
                 feed_dict=feed_dict)
        print("Valid loss: ", valid_loss)
        print("Valid accuracy: ", valid_accuracy)
        print()


def build_model(inputs, labels, num_classes):
    weight_decay = config['weight_decay']
    conv11sz = 16
    conv12sz = 32
    fc3sz = 512
    with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs, conv11sz, scope='conv1')
        net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
        net = layers.convolution2d(net, conv12sz, scope='conv2')
        net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn = tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')

    logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, loss, accuracy


if __name__ == '__main__':
    DATA_DIR = '/home/juraj/PycharmProjects/DubokoUcenje_lab2/data_tf/'
    SAVE_DIR = '/home/juraj/PycharmProjects/DubokoUcenje_lab2/save_tf/'
    node_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    node_y = tf.placeholder(tf.float32, [None, 10])

    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['weight_decay'] = 1e-3
    config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

    # np.random.seed(100)
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    num = train_x[1]
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    with tf.device('/device:GPU:0'):
        sesh = tf.Session()
        logits, loss, accuracy = build_model(node_x, node_y, 10)
        train(train_x, train_y, valid_x, valid_y)
        test(test_x, test_y)
        sesh.close()
