import numpy as np
import data
import matplotlib.pyplot as plt


def logreg_decfun(W, b):
    def classify(X):
        probs = logreg_classify(X, W, b)
        return np.argmax(probs, axis=1)
    return classify


def logreg_classify(X, W, b):
    N = X.shape[0]
    C = max(Y_) + 1
    scores = np.dot(X, W) + b
    expscores = np.exp(scores)
    sumexp = np.sum(expscores, axis=1)
    return np.array([[expscores[i, j] / sumexp[i] for j in range(C)] for i in range(N)])


def logreg_train(X, Y_):
    param_niter = 1000
    param_delta = 0.001
    N = X.shape[0]
    C = max(Y_) + 1
    W = np.random.randn(X.shape[1], C)
    b = np.zeros((1, C))
    for i in range(param_niter):
        scores = np.dot(X, W) + b
        expscores = np.exp(scores)
        sumexp = np.sum(expscores, axis=1)
        probs = np.array([[expscores[i, j]/sumexp[i] for j in range(C)] for i in range(N)])
        logprobs = np.log(probs)
        loss = -np.sum(np.array(
            [[logprobs[i, Y_[i]]] for i in range(N)]
        ))
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        dL_ds = probs - np.array([[1 if Y_[i] == j else 0 for j in range(C)] for i in range(N)])
        grad_W = np.dot(dL_ds.transpose(), X).transpose()
        grad_b = np.sum(dL_ds.transpose(), axis=1)
        W += -param_delta * grad_W
        b += -param_delta * grad_b
    return W, b


if __name__ == "__main__":
    np.random.seed(100)
    G = data.Random2DGaussian()
    N = 100
    C = 3
    X, Y_ = data.sample_gauss(C, N)
    W, b = logreg_train(X, Y_)
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(accuracy, pr, M)

    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
