import numpy as np

from DataVisualizer import DataVisualizer
from ReweighingExperiment import ReweighingExperiment

class ReweighingAnalysisVisualizer(DataVisualizer):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(plt_type = 'reweighted', legend_name = 'Group', parameters = self.parameters)
        self.domain, self.feature = np.linspace(0, 2, 5), 'base_rate'
        self.reweighing = ReweighingExperiment(self.parameters, self.domain)

    def visualize_base_rates(self, df, save = ''):
        results, plots = self.reweighing.summary(self.feature), []
        for (name, adj), result in results.items():
            plot = [' '.join([name, adj]).strip()]
            for i, arr in enumerate(map(np.array, list(zip(*result)))):
                color = 'black' if i else 'green'
                label = 'A = {}'.format(i)
                plot.append((self.domain,) + (arr, color, label))
            plots.append(plot)
        y_label = self.feature.replace('_', ' ').title()
        self.visualize(('Treatment Assignment Bias', y_label), df, *plots, save=save)