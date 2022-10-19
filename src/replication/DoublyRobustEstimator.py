import numpy as np
import pandas as pd

class DoublyRobustEstimator():
    def __init__(self, parameters):
        self.parameters = parameters
        self.num_points = 100
        self.n_bins = 20

    def calibration_curve(self, df, method):
        df.sort_values(method, inplace=True)
        df['bin'] = pd.qcut(df[method], self.n_bins)
        avg_scores, calibrations = [], []
        for _, _df in df.groupby('bin'):
            avg_score = _df[method].mean()
            est = self._estimate(_df)
            delta = 1.96*est.std()/np.sqrt(len(est))
            avg_scores.append(avg_score)
            calibrations.append(est.mean())
        return np.array(calibrations), np.array(avg_scores)

    def precision_recall_curve(self, df, method):
        return 1 - self.false_negative_rate(df, method), self.precision(df, method)

    def roc_curve(self, df, method):
        return self.false_positive_rate(df, method), 1 - self.false_negative_rate(df, method)

    def false_negative_rate(self, df, method):
        domain = np.linspace(0, 1, self.num_points).reshape(-1, 1)
        pred_label = 1 - (df[method].values >= domain).astype(int)
        y_true = self._estimate(df)
        return np.mean(self._numerator(pred_label, y_true), axis = 1) / np.array(y_true).mean()

    def false_positive_rate(self, df, method):
        domain = np.linspace(0, 1, self.num_points).reshape(-1, 1)
        pred_label = (df[method].values >= domain).astype(int)
        y_true = 1 - self._estimate(df)
        return np.mean(self._numerator(pred_label, y_true), axis = 1) / np.array(y_true).mean()

    def precision(self, df, method):
        precisions = []
        for t in np.linspace(0, 1, self.num_points):
            _df = df[df[method] >= t]
            y_true = self._estimate(_df)
            precisions.append(y_true.mean())
        return np.array(precisions)

    def _numerator(self, pred_label, y_true):
        return np.multiply(pred_label, np.array(y_true)[np.newaxis, :])

    def _estimate(self, df):
        treat_num = self.parameters['treat']['name']
        observational = self.parameters['target']['observational']
        return (1 - df[treat_num]) / (1 - df['propensity']) * (df[observational] - df['counterfactual']) + df['counterfactual']