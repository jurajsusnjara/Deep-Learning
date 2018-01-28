import tensorflow as tf
import numpy as np
import data
import matplotlib.pyplot as plt


class TFLogreg:
    def __init__(self, D, C, param_delta=0.5, param_lambda=2e-3):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        # definicija podataka i parametara:
        # definirati self.X, self.Yoh_, self.W, self.b
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = tf.Variable(tf.random_normal([D, C], mean=0, stddev=1/2))
        self.b = tf.Variable(tf.zeros([C]))

        # formulacija modela: izračunati self.probs
        #   koristiti: tf.matmul, tf.nn.softmax
        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

        # formulacija gubitka: self.loss
        # koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        a = self.Yoh_*tf.log(self.probs)
        # self.loss = -tf.reduce_mean(tf.reduce_sum(a, [1]))
        self.loss = -tf.reduce_mean(tf.reduce_sum(a, [1])) + param_lambda * (tf.reduce_sum(self.W**2))

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        self.trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = self.trainer.minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        self.sesh = tf.Session()

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        #   koristiti: tf.initialize_all_variables
        self.sesh.run(tf.initialize_all_variables())

        # optimizacijska petlja
        #   koristiti: tf.Session.run
        for i in range(param_niter):
            loss, _ = self.sesh.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % 100 == 0:
                print(i, loss)

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   koristiti: tf.Session.run
        return self.sesh.run([self.probs], feed_dict={self.X: X})[0]


if __name__ == "__main__":
    def ints_to_one_hots(ints, dimension=None):
        if dimension is None:
            dimension = np.max(ints) + 1
        return np.eye(dimension)[ints]

    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss(4, 100)
    # X, Y_ = data.sample_gmm(4, 2, 30)
    Yoh_ = ints_to_one_hots(Y_)

    # izgradi graf:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre:
    tflr.train(X, Yoh_, 1000)

    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)

    # ispiši performansu (preciznost i odziv po razredima)
    Y = np.argmax(probs, axis=1)
    accuracy, rp, M = data.eval_perf_multi(Y, Y_)
    print("A={}, (R,P)={}, C=\n{}".format(accuracy, rp, M))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: np.argmax(tflr.eval(X), axis=1)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
