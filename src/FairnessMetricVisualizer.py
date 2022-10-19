from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve

from src.DataVisualizer import DataVisualizer
from src.DoublyRobustEstimator import DoublyRobustEstimator

class FairnessMetricVisualizer(DataVisualizer):
    def __init__(self, metric, parameters):
        self.metric = metric
        self.parameters = parameters
        if self.metric != 'calibration':
            super().__init__(plt_type = self.metric, legend_name = 'Model', parameters = self.parameters)
        else:
            super().__init__(plt_type = self.metric, legend_name = 'Model', parameters = self.parameters, n_bins = 20)

        self.treat_num = self.parameters['treat']['name']
        self.observational = self.parameters['target']['observational']
        self.counterfactual = self.parameters['target']['counterfactual']
        self.dr = DoublyRobustEstimator(parameters=self.parameters)
        self.functions = {
            'roc': (roc_curve, self.dr.roc_curve),
            'precision_recall': (precision_recall_curve, self.dr.precision_recall_curve), 
            'calibration': (self._calibration, self.dr.calibration_curve)
        }
        self.titles = [
            "Observational Evaluation",
            "Control",
            "Doubly-Robust",
            "True Counterfactual"
        ]
        self.axis_labels = {
            'roc': ('False Positive Rate', "Recall"),
            'precision_recall': ("Recall", "Precision"),
            'calibration': ('Average Risk Score', "Outcome Rate")
        }
        self.colors = ['tab:blue', 'tab:orange']
        self.methods = ['Observational', 'Counterfactual']

    # metric = key in self.functions
    def visualize_metric(self, df, save = ''):
        function, doubly_robust = self.functions[self.metric]
        figure = [self.axis_labels[self.metric], df]
        for title in self.titles:
            plot = [title]
            for color, method in zip(self.colors, self.methods):
                _df = df.copy()
                score = method.lower()
                if title == 'Control':
                    _df = _df[_df[self.treat_num] == 0]
                if title != 'Doubly-Robust':
                    column = self.counterfactual if title == "True Counterfactual" else self.observational
                    if self.metric == 'roc':
                        x, y, _ = function(*_df[[column, score]].values.T)
                    elif self.metric == 'precision_recall':
                        y, x, _ = function(*_df[[column, score]].values.T)
                    else:
                        y, x = function(*_df[[column, score]].values.T)
                else:
                    if self.metric == 'calibration':
                        y, x = doubly_robust(_df, score)
                    else:
                        x, y = doubly_robust(_df, score)
                plot.append((x, y, color, method))
            figure.append(plot)
        self.visualize(*figure, save = save)

    def _calibration(self, x, y):
        return calibration_curve(x, y, n_bins=20, strategy = 'quantile')