import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import skimage as ski
import skimage.io
import math
import matplotlib.pyplot as plt
import time


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def build_model(inputs, labels, num_classes):
    weight_decay = 1e-3
    conv11sz = 16
    conv12sz = 32
    fc3sz = 256
    fc4sz = 128
    fc5sz = 10

    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, conv11sz, scope='conv1')
        net = layers.max_pool2d(net, kernel_size=3, stride=2, scope='pool1')
        net = layers.convolution2d(net, conv12sz, scope='conv2')
        net = layers.max_pool2d(net, kernel_size=3, stride=2, scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')
        net = layers.fully_connected(net, fc4sz, scope='fc4')
        net = layers.fully_connected(net, fc5sz, scope='fc5')

    logits = layers.fully_connected(net, num_classes, activation_fn=None,
                                    weights_regularizer=layers.l2_regularizer(weight_decay), scope='logits')
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)

    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    return logits, loss, accuracy, losses


def evaluate(y, y_predicted):
    N, C = y.shape
    confusion_matrix = np.zeros(shape=(C, C))
    for i in range(0, N):
        x = np.argmax(y[i])
        y = np.argmax(y_predicted[i])
        confusion_matrix[x, y] += 1

    precision = np.zeros(shape=C)
    recall = np.zeros(shape=C)
    tp_micro = 0
    tn_micro = 0
    for i in range(0, confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        tp_micro += tp
        fp = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i]) - confusion_matrix[i, i]
        tn_micro += N - tp - fp - fn
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
    accuracy = (tp_micro + tn_micro) / (N * C)

    return accuracy, confusion_matrix, precision, recall


def labels_to_one_hot(Y, C):
    Yoh_ = np.zeros((Y.shape[0], C))
    Yoh_[range(Y.shape[0]), Y] = 1
    return Yoh_


def train(train_x, train_y, valid_x, valid_y):
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []

    sess.run(tf.initialize_all_variables())
    conv1_var = tf.contrib.framework.get_variables('conv1')[0]
    conv1_weights = conv1_var.eval(session=sess)
    draw_conv_filters(0, 0, conv1_weights, SAVE_DIR)

    for epoch_num in range(1, num_epochs + 1):
        lr = 0.01 / math.pow(epoch_num, 2)
        trainer = tf.train.GradientDescentOptimizer(lr)
        train_op = trainer.minimize(loss)

        train_x, train_y = shuffle_data(train_x, train_y)
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        cnt_correct = 0
        train_loss_sum = 0
        for step in range(num_batches):
            offset = step * batch_size
            batch_x = train_x[offset:(offset + batch_size), ...]
            batch_y = train_y[offset:(offset + batch_size)]
            feed_dict = {node_x: batch_x, node_y: batch_y}
            start_time = time.time()
            run_ops = [train_op, loss, logits]
            ret_val = sess.run(run_ops, feed_dict=feed_dict)
            _, loss_val, logits_val = ret_val

            yp = np.argmax(logits_val, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()
            train_loss_sum += loss_val

            duration = time.time() - start_time
            if (step + 1) % 50 == 0:
                sec_per_batch = float(duration)
                format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
                print(format_str % (epoch_num, step + 1, num_batches, loss_val, sec_per_batch))

        print('Train error:')
        train_acc = cnt_correct / num_examples * 100
        train_loss = train_loss_sum / num_batches
        print("Train accuracy = %.2f" % train_acc)
        print("Train loss = %.2f" % train_loss)

        print('Validation error:')
        feed_dict = {node_x: valid_x, node_y: valid_y}
        valid_loss, valid_acc = sess.run([loss, accuracy], feed_dict=feed_dict)
        print("Valid accuracy = %.2f" % valid_acc)
        print("Valid loss = %.2f" % valid_loss)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [valid_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [valid_acc]
        plot_data['lr'] += [lr]
        plot_training_progress(SAVE_DIR, plot_data)

        conv1_var = tf.contrib.framework.get_variables('conv1')[0]
        conv1_weights = conv1_var.eval(session=sess)
        draw_conv_filters(epoch_num, num_batches, conv1_weights, SAVE_DIR)


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


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
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def get20FalseClassificationPictures(test_x, test_y, test_losses, test_logits):
    for counter in range(1, 21):
        index = np.argmax(test_losses)
        draw_image(test_x[index], data_mean, data_std)
        top3 = []
        logit = test_logits[index]
        for i in range(0, 3):
            max = np.argmax(logit)
            top3.insert(i, max)
            logit[max] = -999999

        format_str = 'Figure %d, loss = %.2f, top 3 class = %d, %d, %d, true class = %d'
        print(format_str % (counter, test_losses[index], top3[0], top3[1], top3[2], np.argmax(test_y[index])))

        test_losses[index] = -999999


DATA_DIR = '/home/juraj/PycharmProjects/DubokoUcenje_lab2/cifar-10-batches-py/'
SAVE_DIR = '/home/juraj/PycharmProjects/DubokoUcenje_lab2/save_cifar/'

num_epochs = 8
batch_size = 50
save_dir = SAVE_DIR

img_height = 32
img_width = 32
num_channels = 3

if __name__ == '__main__':
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_y = labels_to_one_hot(train_y, 10)
    valid_y = labels_to_one_hot(valid_y, 10)
    test_y = labels_to_one_hot(test_y, 10)

    node_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    node_y = tf.placeholder(tf.float32, [None, 10])
    sess = tf.Session()
    logits, loss, accuracy, losses = build_model(node_x, node_y, 10)

    train(train_x, train_y, valid_x, valid_y)

    print('Test error:')
    feed_dict = {node_x: test_x, node_y: test_y}
    test_loss, test_losses, test_acc, test_logits = sess.run([loss, losses, accuracy, logits], feed_dict=feed_dict)
    print("Test accuracy = %.2f" % test_acc)
    print("Test loss = %.2f" % test_loss)

    get20FalseClassificationPictures(test_x, test_y, test_losses, test_logits)
