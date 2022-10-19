import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from fairlearn.metrics import MetricFrame

from src.DataVisualizer import DataVisualizer

class EqualizedOddsPostProcessingAnalysis(DataVisualizer):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(plt_type = 'roc', legend_name = 'Group', parameters = self.parameters)
        self.methods = ['Original', 'Post-Processed']
        self.columns = ['counterfactual', 'eo_fair_pred']
        self.colors = ['black', 'green']
        self.metrics = ('False Positive Rate', "Recall")

        self.sensitive = self.parameters['features']['sensitive']
        self.observational = self.parameters['target']['observational']
        self.counterfactual = self.parameters['target']['counterfactual']
        self.dataset = {}
        self.errors = []

    def visualize_roc(self, df, save = ''):
        self._generate_dataset(df)
        original = ['Original']
        post_processed = ['Post-Processed']
        for k, v in self.dataset.items():
            method, group, color = k
            content = tuple(v.values()) + (color, group)
            if method == 'Original':
                original.append(content)
            else:
                post_processed.append(content)
        return self.visualize(self.metrics, df, original, post_processed, save = save)

    def error_analysis(self, df):
        for adj in ['observational', 'counterfactual']:
            self.errors.append(self._construct_metricframe(df, adj).by_group)
        return self._postprocess_df(self.errors)

    def _generate_dataset(self, df):
        for method, column in zip(self.methods, self.columns):
            for i, _df in df.groupby([self.parameters['features']['sensitive']]):
                group = 'A = 0' if not i else 'A = 1'
                fpr, tpr, _ = roc_curve(_df[self.counterfactual], _df[column])
                self.dataset[(method, group, self.colors[i])] = {'fpr': fpr, 'tpr': tpr}

    def _postprocess_df(self, metricframes):
        errors = pd.concat(metricframes, axis = 1)
        errors.reset_index(inplace=True)
        errors['Method'] = errors['Method'].map({False: 'Original', True: 'Post-Processed'})
        errors = errors.sort_values(['Method', 'Group'], ascending=[True, False])
        errors = errors[['Group', 'Method', 'cGFNR', 'cGFPR', 'oGFNR', 'oGFPR']]
        return errors

    def _construct_metricframe(self, df, adj):
        ## adj = ['observational', 'counterfactual']
        _df = self._preprocess_df(df)
        metrics = {
            adj[0] + "GFNR": self._compute_fnr_cost_obs,
            adj[0] + "GFPR": self._compute_fpr_cost_obs
        }
        metricframe = MetricFrame(
            metrics=metrics,
            y_true=_df[self.observational] if adj == 'observational' else _df[self.counterfactual],
            y_pred=_df['counterfactual'],
            sensitive_features=_df[['Method', 'Group']]
        )
        return metricframe

    def _preprocess_df(self, df):
        _df = df.melt(id_vars=df.columns.difference(self.columns, sort=False), 
            var_name="Method", value_name="score")

        _df['Method'] = _df['Method'].map({'counterfactual': False, 'eo_fair_pred': True})
        _df[self.sensitive] = _df[self.sensitive].map({0: 'A = 0', 1: 'A = 1'})
        _df = _df.rename(columns = {'score': 'counterfactual', self.sensitive: 'Group'})
        return _df

    def _compute_fpr_cost_obs(self, label, pred):
        return np.mean(pred[label == 0])

    def _compute_fnr_cost_obs(self, label, pred):
        return 1 - np.mean(pred[label == 1])