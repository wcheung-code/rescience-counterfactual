import numpy as np
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import true_negative_rate, false_negative_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import selection_rate
from sklearn.metrics import confusion_matrix

class MetricFrameGenerator():
    def __init__(self, weights = None):
        self.weights = weights
        self.params = {
            'base_rate': {'sample_weight': self.weights},
            'tnr': {'sample_weight': self.weights},
            'fpr': {'sample_weight': self.weights},
            'fnr': {'sample_weight': self.weights},
            'tpr': {'sample_weight': self.weights},
        }
        self.metrics = {
            "base_rate": self._base_rate,
            "tnr": self._true_negative_rate,
            "fpr": self._false_positive_rate,
            "fnr": self._false_negative_rate,
            "tpr": self._true_positive_rate,
        }

    def generate(self, y_true, y_pred, sensitive_features):
        train_metricframe = MetricFrame(
            metrics=self.metrics,
            y_true=y_true, #train['Y'],
            y_pred=y_pred, #train['counterfactual'],
            sensitive_features=sensitive_features, #train['A']
            sample_params=self.params
        )
        return train_metricframe.by_group 

    ## Extension:
    # def _true_negative_rate(self, y_true, y_pred, sample_weight=None):
    #     return true_negative_rate(y_true, y_pred.round())

    # def _false_positive_rate(self, y_true, y_pred, sample_weight=None):
    #     return false_positive_rate(y_true, y_pred.round())

    # def _false_negative_rate(self, y_true, y_pred, sample_weight=None):
    #     return false_negative_rate(y_true, y_pred.round())

    # def _true_positive_rate(self, y_true, y_pred, sample_weight=None):
    #     return true_positive_rate(y_true, y_pred.round())

    # ## Generalize fairlearn/fairlearn/metrics/_extra_metrics.py to add normalize as parameter
    def _true_negative_rate(self, y_true, y_pred, sample_weight=None):
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_true, y_pred.round(), sample_weight=sample_weight, labels=[0, 1], normalize="all").ravel()
        return tnr

    def _false_positive_rate(self, y_true, y_pred, sample_weight=None):
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_true, y_pred.round(), sample_weight=sample_weight, labels=[0, 1], normalize="all").ravel()
        return fpr

    def _false_negative_rate(self, y_true, y_pred, sample_weight=None):
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_true, y_pred.round(), sample_weight=sample_weight, labels=[0, 1], normalize="all").ravel()
        return fnr

    def _true_positive_rate(self, y_true, y_pred, sample_weight=None):
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_true, y_pred.round(), sample_weight=sample_weight, labels=[0, 1], normalize="all").ravel()
        return tpr

    def _base_rate(self, y_true, y_pred, sample_weight=None):
        return np.average(y_true, weights=sample_weight)