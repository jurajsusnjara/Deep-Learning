import data
import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    return np.exp(x)/(1 + np.exp(x))


def binlogreg_classify(X, w, b):
    return softmax(np.dot(X, w) + b)


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


def binlogreg_train(X, Y_):
    param_niter = 1000
    param_delta = 0.001
    N = X.shape[0]
    w = np.random.randn(X.shape[1], 1)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = softmax(scores)
        loss = -np.sum(np.log(np.array(
            [[1 - probs[i, 0]] if Y_[i] == 0 else [probs[i, 0]] for i in range(Y_.shape[0])])))
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        dL_scores = np.array(
            [[(probs[i, 0] - 1)] if Y_[i] == 1 else [probs[i, 0]] for i in range(Y_.shape[0])])
        grad_w = np.dot(dL_scores.transpose(), X).transpose()
        grad_b = np.sum(dL_scores)/N
        w += -param_delta * grad_w
        b += -param_delta * grad_b
    return w, b


if __name__ == "__main__":
    np.random.seed(100)
    G = data.Random2DGaussian()
    N = 100
    C = 2
    X, Y_ = data.sample_gauss(C, N)
    w, b = binlogreg_train(X, Y_)
    probs = binlogreg_classify(X, w, b)
    Y = np.array([1 if probs[i, 0] > 0.5 else 0 for i in range(probs.shape[0])])
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_)
    print(accuracy, recall, precision, AP)

    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
