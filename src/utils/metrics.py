import numpy as np

from sklearn.metrics import f1_score


def accuracy_metric(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    return np.mean(preds == labels)


def f1_metric(preds, labels):
    return f1_score(labels, preds)