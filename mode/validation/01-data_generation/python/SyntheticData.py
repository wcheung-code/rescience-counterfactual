import numpy as np
import pandas as pd

class SyntheticData():
    def __init__(self, treatment_effect, treatment_assignment_bias, seed = 0):
        self.seed = seed
        np.random.seed(self.seed)
        self.treatment_effect = treatment_effect
        self.treatment_assignment_bias = treatment_assignment_bias

    def generate(self, num_points, treatment_as_feature = False):

        self.columns = {
            'treatment': 'treat_num',
            'outcome': 'outcome',
            'features': ['Z', 'A', 'treat_num'],
            'sensitive_feature': 'A',
            'observational_response': 'Y',
            'counterfactual_response': 'Y_0',
            'alternative_response': 'Y_1'
        }

        Z = np.random.normal(loc = 0, scale = 1, size = num_points)
        A = np.random.binomial(n = 1, p = 0.5, size = num_points)
        Y_0 = np.random.binomial(n = 1, p = self._sigmoid(Z - 0.5))
        Y_1 = np.random.binomial(n = 1, p = self.treatment_effect * self._sigmoid(Z - 0.5))
        T = np.random.binomial(n = 1, p = self._sigmoid(Z - 0.5 +
            self.treatment_assignment_bias*A))
        Y = T*Y_1 + (1-T)*Y_0

        self.config = {
            'treat': {
                'name': self.columns['treatment'], # column name of treatment data 
                ## control group = 0
            },
            'outcome': {
                'name': self.columns['outcome'], # column name of outcome data
            },
            'features': {
                'training': list(set(self.columns['features']) - set([self.columns['treatment'] if not treatment_as_feature else ''])),
                'sensitive': self.columns['sensitive_feature'],
            }, # predictors used for training model
            'target': {
                'observational': self.columns['observational_response'],
                'counterfactual': self.columns['counterfactual_response'],
            }, # response variable for training model
            'is_train': False, # identifier that determines if row is part of training data
            'sample_weight': None # presence of sample weights
        }

        synthetic_data = dict(zip(self.columns['features'], [Z, A, T]))
        synthetic_data.update(
            {
                self.columns['observational_response']: Y,
                self.columns['alternative_response']: Y_1,
                self.columns['counterfactual_response']: Y_0,
                self.columns['outcome']: np.where(Y == 1, 'harm', 'ok'),
            }
        )

        df = pd.DataFrame(data=synthetic_data)
        return df, self.config

    def _sigmoid(self, z):
        return np.reciprocal(1 + np.exp(-z))