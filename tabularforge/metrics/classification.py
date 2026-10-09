"""Classification metrics."""
import math


def accuracy(preds, truths):
    if not truths:
        return 1.0
    return sum(1 for p, t in zip(preds, truths) if p == t) / len(truths)


def _confusion(preds, truths):
    tp = fp = fn = 0
    labels = set(truths)
    for p, t in zip(preds, truths):
        if p == t:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def precision(preds, truths, average="micro"):
    if not truths:
        return 0.0
    tp, fp, fn = _confusion(preds, truths)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(preds, truths, average="micro"):
    if not truths:
        return 0.0
    tp, fp, fn = _confusion(preds, truths)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(preds, truths, average="micro"):
    p = precision(preds, truths)
    r = recall(preds, truths)
    return 2 * p * r / (p + r) if (p + r) else 0.0
