import tensorflow as tf
import numpy as np
import data
import matplotlib.pyplot as plt


class TFDeep:
    def __init__(self, layer_dimensions, param_delta=0.1, param_lambda=1e-4):

        self.X = tf.placeholder(tf.float32, [None, layer_dimensions[0]])
        self.Yoh_ = tf.placeholder(tf.float32, [None, layer_dimensions[-1]])
        self.Ws = []
        self.bs = []
        self.hs = []

        h = self.X
        for d in layer_dimensions[1:-1]:
            indim = h.shape[1].value
            W = tf.Variable(tf.random_normal([indim, d], mean=0, stddev=1/indim),
                            name="W_" + str(len(self.Ws) + 1))
            b = tf.Variable(tf.zeros([d]),
                            name="b_" + str(len(self.bs) + 1))
            self.Ws.append(W)
            self.bs.append(b)
            h = tf.nn.sigmoid(tf.matmul(h, W) + b)
            self.hs.append(h)

        indim = h.shape[1].value
        W = tf.Variable(tf.random_normal([indim, layer_dimensions[-1]], mean=0, stddev=1/indim),
                        name="W_" + str(len(self.Ws) + 1))
        b = tf.Variable(tf.zeros([layer_dimensions[-1]]),
                        name="b_" + str(len(self.bs) + 1))
        self.Ws.append(W)
        self.bs.append(b)

        self.probs = tf.nn.softmax(tf.matmul(h, W) + b)

        a = self.Yoh_*tf.log(self.probs)
        self.loss = -tf.reduce_mean(tf.reduce_sum(a, [1]))
        for W in self.Ws:
            self.loss += param_lambda*tf.reduce_sum(W**2)

        self.trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = self.trainer.minimize(self.loss)

        self.sesh = tf.Session()

    def train(self, X, Yoh_, param_niter):
        self.sesh.run(tf.initialize_all_variables())
        for i in range(param_niter):
            loss, _ = self.sesh.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 500 == 0:
                print(i, loss)

    def eval(self, X):
        return self.sesh.run([self.probs], feed_dict={self.X: X})[0]

    def count_params(self):
        count = 0
        for v in tf.trainable_variables():
            print(v.name, v.shape)
            c = v.shape[0].value
            for b in v.shape[1:]:
                c *= b.value
            count += c
        print("parameter count = "+str(count))


if __name__ == "__main__":
    def ints_to_one_hots(ints, dimension=None):
        if dimension is None:
            dimension = np.max(ints) + 1
        return np.eye(dimension)[ints]

    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    # X, Y_ = data.sample_gauss(3, 100)
    X, Y_ = data.sample_gmm(6, 2, 10)
    Yoh_ = ints_to_one_hots(Y_)

    # izgradi graf:
    layers = [X.shape[1], 10, 10, Yoh_.shape[1]]
    tf_deep = TFDeep(layers)

    # nauči parametre:
    tf_deep.train(X, Yoh_, 10000)

    tf_deep.count_params()

    # dohvati vjerojatnosti na skupu za učenje
    probs = tf_deep.eval(X)

    # ispiši performansu (preciznost i odziv po razredima)
    Y = np.argmax(probs, axis=1)
    accuracy, rp, M = data.eval_perf_multi(Y, Y_)
    print("A={}, (R,P)={}, C=\n{}".format(accuracy, rp, M))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: tf_deep.eval(X)[:,1]#np.argmax(tf_deep.eval(X), axis=1)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
