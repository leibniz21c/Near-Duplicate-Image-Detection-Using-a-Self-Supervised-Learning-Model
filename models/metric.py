from torchmetrics.functional import confusion_matrix
from torch import transpose
import torch.nn.functional as F

def cosine_distance(x1, x2):
    """Cosine distance"""
    return 1 - F.cosine_similarity(x1, x2)

def true_positive(prediction, truth):
    """Returns the number of true positive for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - True and True (True Positive)
    """
    return int(confusion_matrix(prediction, truth, num_classes=2)[1, 1])


def true_negative(prediction, truth):
    """Returns the number of true negative for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - False and False (True Negative)
    """
    return int(confusion_matrix(prediction, truth, num_classes=2)[0, 0])


def false_positive(prediction, truth):
    """Returns the number of false positive for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - True and False (False Positive)
    """
    return int(confusion_matrix(prediction, truth, num_classes=2)[0, 1])


def false_negative(prediction, truth):
    """Returns the number of true negative for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - False and True (False Negative)
    """
    return int(confusion_matrix(prediction, truth, num_classes=2)[1, 0])


def precision(prediction, truth):
    """Metric 1. Precision(PPV)
    :(Precision) = \frac{TP}{TP + FP}
    : In some rare cases, the calculation of Precision or Recall can cause a division by 0.
    : See this https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    """
    try:
        return true_positive(prediction, truth) / (
            true_positive(prediction, truth) + false_positive(prediction, truth)
        )
    except:
        return 1.0


def recall(prediction, truth):
    """Metric 2. Recall(Sensitivity, true positive rate)
    :(Recall) = \frac{TP}{TP + FN}
    : In some rare cases, the calculation of Precision or Recall can cause a division by 0.
    : See this https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    """
    try:
        return true_positive(prediction, truth) / (
            true_positive(prediction, truth) + false_negative(prediction, truth)
        )
    except:
        return 1.0


def f1_score(prediction, truth):
    """Metric 3. F1-score
    :(F1-score) = 2 * \frac{Precision * Recall}{Precision + Recall}
    : In some rare cases, the calculation of Precision or Recall can cause a division by 0.
    : See this https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    """
    rec = recall(prediction, truth)
    prec = precision(prediction, truth)
    try:
        return 2 * prec * rec / (prec + rec)
    except:
        return 1.0


def false_positive_rate(prediction, truth):
    """Metric 4. False positive rate
    :(False positive rate) = \frac{FP}{FP + TN}
    """
    try:
        return 1.0 - true_negative(prediction, truth) / (
            true_negative(prediction, truth) + false_positive(prediction, truth)
        )
    except:
        return 0.0
