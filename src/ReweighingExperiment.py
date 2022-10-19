import numpy as np
from collections import defaultdict

from src.SyntheticData import SyntheticData
from src.MetricFrameGenerator import MetricFrameGenerator

class ReweighingExperiment():
    def __init__(self, parameters, domain):
        self.domain = domain
        self.parameters = parameters
        self.sensitive = self.parameters['features']['sensitive']
        self.observational = self.parameters['target']['observational']
        self.names = ['', 'Reweighted']
        self.columns = self.parameters['target'].values()
        self.adjectives = self.parameters['target'].keys()

    def summary(self, feature):
        reweighed = defaultdict(list)
        for t in self.domain: 
            synthetic_data = SyntheticData(treatment_effect = 0.1, 
                                        treatment_assignment_bias = t,
                                        seed = 1)
            __df, _ = synthetic_data.generate(num_points = 100000)
            metric_frame =  MetricFrameGenerator()
            weights = self._compute_weights(__df)
            for (a, y), weight in np.ndenumerate(weights):
                __df.loc[(__df[self.observational] == 1-y) & (__df[self.sensitive] == 1-a), 'weight'] = weight
            weighted_metric_frame =  MetricFrameGenerator(weights = __df['weight'])

            frames = [metric_frame, weighted_metric_frame]
            for frame, name in zip(frames, self.names):
                for col, adj in zip(self.columns, self.adjectives):
                    reweighed[(name, adj)].append(frame.generate(
                        __df[col], __df[col], __df[self.sensitive]).loc[:, feature])
        return reweighed

    def _compute_weights(self, __df):
        criteria = [self.sensitive, self.observational]
        counts = __df.groupby(criteria).count().iloc[:, 0].to_numpy()[::-1].reshape(2,2).astype(float)
        A = np.outer(*np.c_[__df[criteria].mean(), (1 - __df[criteria].mean())])
        return np.multiply(A, len(__df) * np.reciprocal(counts))