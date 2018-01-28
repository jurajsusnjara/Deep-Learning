import numpy as np


def output_softmax(o):
    # o shape = (seq, minibatch, dim)
    res = np.zeros_like(o)
    for seq_idx in range(o.shape[0]):
        for row_idx in range(o.shape[1]):
            row = o[seq_idx][row_idx]
            sm = np.exp(row) / np.sum(np.exp(row), axis=0)
            res[seq_idx][row_idx][:] = sm
    return res


y = np.random.randint(0,2,size=(2, 3, 5))
yhat = output_softmax(np.random.random(size=(2,3,5)))
print(y)
# print(yhat)
# lg = np.log(yhat)
# res = y*lg
# print('res')
# print(res)
# print('loss')
print(np.sum(y, axis=2))
