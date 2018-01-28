import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.array([[0,0,0],[1,1,1],[2,2,2],[0,3,3]])
Y = np.array([1,2,3,4])

# fit the model
clf = svm.SVC()
clf.fit(X, Y)