import numpy as np
import data


def softmax(x):
    return np.exp(x)/(1 + np.exp(x))


def fcann2_train(X, Y_):
    param_niter = 1000
    param_delta = 0.001
    param_lambda = 1e-3
    hidden_no = 5
    D = X.shape[1]
    N = X.shape[0]
    C = max(Y_) + 1
    W1 = np.random.randn(D, hidden_no)
    b1 = np.random.randn(1, hidden_no)
    W2 = np.random.randn(hidden_no, C)
    b2 = np.random.randn(1, C)
    for i in range(param_niter):
        s1 = np.dot(X, W1) + b1
        H1 = np.maximum(s1, 0)
        s2 = np.dot(H1, W2) + b2
        P = softmax(s2)
        logP = np.log(P)
        loss = -np.sum(np.array([[logP[i, Y_[i]]] for i in range(N)]))
        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, loss))
        G_s2 = P - np.array([[1 if Y_[i] == j else 0 for j in range(C)] for i in range(N)])
        grad_W2 = np.dot(np.transpose(G_s2), H1)
        grad_b2 = np.sum(G_s2.transpose(), axis=1)
        G_s1 = np.array([G_s2[i, :].dot(W2.transpose()).dot(np.diag(s1[i, :] > 0)) for i in range(N)])
        grad_W1 = G_s1.transpose().dot(X)
        grad_b1 = np.sum(G_s1.transpose(), axis=1)
        W1 += -param_delta * grad_W1.transpose()
        b1 += -param_delta * grad_b1
        W2 += -param_delta * grad_W2.transpose()
        b2 += -param_delta * grad_b2
    return W1, b1, W2, b2


def fcann2_classify(X, W, b):
    pass


if __name__ == '__main__':
    np.random.seed(100)
    X,Y_ = data.sample_gmm(4, 2, 30)
    fcann2_train(X, Y_)
