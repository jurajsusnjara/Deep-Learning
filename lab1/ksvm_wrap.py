import numpy as np
from sklearn.svm import SVC
import data
import matplotlib.pyplot as plt


class KSVMWrap():
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_


if __name__ == '__main__':
    np.random.seed(100)

    d = 2
    X, Y_ = data.sample_gmm(6, 2, 10)

    svm = KSVMWrap(X, Y_)
    support = svm.support()
    print(str(len(support)) + " support vectors")

    Y = svm.predict(X)
    scores = svm.get_scores(X)
    print(scores.shape)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, rp, M = data.eval_perf_multi(Y, Y_)
    print("A={}, (R,P)={}, C=\n{}".format(accuracy, rp, M))

    print(svm.support())

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda X: svm.get_scores(X) if d == 2 else svm.predict(X)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=support)
    plt.show()