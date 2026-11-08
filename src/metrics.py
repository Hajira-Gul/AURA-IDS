import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix)


def evaluate(y_true, y_pred, labels):
    """Shared evaluation used by every method, so all numbers are computed the same way.

    y_true, y_pred: arrays of class labels (strings or ints)
    labels: list of classes to score (macro metrics average over these only)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro",
                        zero_division=0)
    macro_prec = precision_score(y_true, y_pred, labels=labels, average="macro",
                                 zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None,
                                    zero_division=0)

    # per-class false positive rate = FP / (FP + TN), then macro-averaged
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fpr = float(np.mean(fp / np.maximum(fp + tn, 1)))

    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "per_class_recall": dict(zip(labels, per_class_recall.round(4).tolist())),
        "fpr": fpr,
        "accuracy": float((y_true == y_pred).mean()),
    }