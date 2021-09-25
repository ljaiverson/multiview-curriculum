from sklearn.metrics import roc_auc_score
import numpy as np

def getAUC(y_true, y_score, task):
    auc = 0
    zero = np.zeros_like(y_true)
    one = np.ones_like(y_true)
    for i in range(y_score.shape[1]):
        y_true_binary = np.where(y_true == i, one, zero)
        y_score_binary = y_score[:, i]
        auc += roc_auc_score(y_true_binary, y_score_binary)
    return auc / y_score.shape[1]

