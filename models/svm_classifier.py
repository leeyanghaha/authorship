from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import utils.key_utils as ku
import os
from sklearn.externals import joblib
import numpy as np



class Svm:
    def __init__(self, penalty, loss, dual, multi_class, fit_intercept, max_iter):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, multi_class=multi_class,
                             fit_intercept=fit_intercept, max_iter=max_iter)

    def fit(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def save(self, model_name):
        path = os.path.join(ku.model_root, model_name)
        if os.path.exists(path):
            print('rm {} and save {}.'.format(path, model_name))
        else:
            joblib.dump(self.clf, path)
            print('save {} at {}.'.format(model_name, path))

def load(model_name):
    path = os.path.join(ku.model_root, model_name)
    if not os.path.exists(path):
        raise ValueError('{} does not exits.'.format(path))
    else:
        return joblib.load(path)






