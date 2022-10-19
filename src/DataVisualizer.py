import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.DoublyRobustEstimator import DoublyRobustEstimator

class DataVisualizer(DoublyRobustEstimator):
    def __init__(self, plt_type, legend_name, parameters, n_bins = None):
        self.parameters = parameters
        self.observational = self.parameters['target']['observational']
        self.counterfactual = self.parameters['target']['counterfactual']
        super().__init__(parameters = self.parameters)
        self.plt_type = plt_type #'calibration', 'roc', 'precision_recall'
        self.legend_name = legend_name
        if self.plt_type == 'calibration':
            self.n_bins = n_bins
        self.plt_settings = {
            "figure.dpi": 72,
            "figure.facecolor": 'white',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.formatter.limits': (0, 1),
            'legend.frameon': False,
        }
    
    def visualize(self, axis_labels, df, *args, save = ''):
        x_axis_label, y_axis_label = axis_labels
        with mpl.rc_context(self.plt_settings):
            num_plots = len(args)
            plt.rcParams["figure.figsize"] = (5 * num_plots, 5)
            fig = plt.figure()
            fig, axes = plt.subplots(1, num_plots)
            for axis, lines in zip(axes.ravel(), args):
                title = lines.pop(0)
                for x, y, color, label in lines:
                    axis.plot(x, y, color = color, label= label)
                    if self.plt_type == 'calibration':
                        self._plot_confidence_intervals(axis, df, x, y, color, label, title)

                padding = 0.05; lower, upper = 0 - padding, (0.5 if self.plt_type == 'reweighted' else 1) + padding
                axis.set_xlim([lower, upper + (1.5 if self.plt_type == 'reweighted' else 0)])
                axis.set_ylim([lower, upper])
                axis.ticklabel_format(style='plain')

                if self.plt_type == 'calibration':
                    
                    _x = np.linspace(lower, upper, 50)
                    _y = np.linspace(lower, upper, 50)
                    axis.plot(_x, _y, color = 'black', linestyle = '--')

                axis.set_title(title)
                if self.plt_type != 'reweighted':
                    axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
                    axis.set_yticks([0, 0.25, 0.5, 0.75, 1])

            fig.text(0.5, 0, x_axis_label, ha='center', va='center')
            fig.text(0, 0.5, y_axis_label, ha='center', va='center', rotation='vertical')

            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2, title=self.legend_name)

            fig.tight_layout();
            if save:
                plt.savefig(save, bbox_inches='tight')
            plt.show()

    def _plot_confidence_intervals(self, axis, df, x, y, color, label, title, alpha = 0.5):
        if title == 'True Counterfactual':
            response = self.counterfactual
        else:
            response = self.observational
        _df = df.copy()
        _df.sort_values(label.lower(), inplace=True)
        _df['bin'] = pd.qcut(_df[label.lower()], self.n_bins)
        y_lower, y_upper = [], []
        for _y, (_, __df) in zip(y, _df.groupby('bin')):
            if title == 'Doubly-Robust':
                est = self._estimate(__df)
                delta =  1.96*est.std()/np.sqrt(len(est))
            else:
                delta =  1.96*__df[response].std()/np.sqrt(len(__df))
            y_lower.append(_y - delta)
            y_upper.append(_y + delta)
        axis.fill_between(x, np.array(y_lower), np.array(y_upper), color = color, alpha = alpha)