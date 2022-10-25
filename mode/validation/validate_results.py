import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.SyntheticData import SyntheticData
from src.SupervisedLearningModel import SupervisedLearningModel
from src.MetricFrameGenerator import MetricFrameGenerator
from src.EqualizedOddsPostProcesser import EqualizedOddsPostProcesser
from src.EqualizedOddsPostProcessingAnalysis import EqualizedOddsPostProcessingAnalysis
from src.FairnessMetricVisualizer import FairnessMetricVisualizer

import pickle

if __name__ == '__main__':

    FIGURE_DIRECTORY = "./validation"

    c, k = 0.1, 1.6
    num_points = 100000

    num_seeds = 5

    model_list = ['propensity', 'observational', 'counterfactual']

    def synthetic_experiment(seed, c = c, k = k, num_points = num_points):
        
        synthetic_data = SyntheticData(
            treatment_effect = c, 
            treatment_assignment_bias = k,
            seed = seed)
        df, config = synthetic_data.generate(num_points = num_points)

        for model_key in model_list:
            params = config.copy()
            clf = LogisticRegression(penalty = 'none')
            model = SupervisedLearningModel(model = clf, name = model_key, seed = seed)
            if model_key == 'propensity':
                params['target'] = params['treat']['name']
            else:
                params['target'] = params['outcome']['name']
            model.fit(df, params)
            train = df[params['features']['training']]
            if model_key == 'propensity':
                df[model_key] = clf.predict_proba(train)[:, 1:]
            else:
                df[model_key] = clf.predict_proba(train)[:, :1]

        ## Postprocess test and training datasets via equalized odds
        metric_frame =  MetricFrameGenerator()
        equalized_odds = EqualizedOddsPostProcesser(config)
        observational = config['target']['observational']
        sensitive = config['features']['sensitive']
        _df, postprocessed = df[df.columns.difference(['treat', 'outcome'], sort=False)], {}
        datasets = train_test_split(_df, test_size=0.3, random_state=0)
        for key, raw_data in zip(['train', 'test'], datasets):
            data = raw_data.copy()
            metricframe = metric_frame.generate(data[observational], data['counterfactual'], data[sensitive])
            if key == 'train':
                mix_rates = equalized_odds.mix_rates(data, metricframe)
            probs = equalized_odds.post_process(data, metricframe, mix_rates)
            for sensitive_class, prob in probs.items():
                data.loc[data[sensitive] == sensitive_class, 'eo_fair_pred'] = prob
            postprocessed[key] = data

        eo_analysis = EqualizedOddsPostProcessingAnalysis(config)
        _eo = eo_analysis.visualize_roc(postprocessed['test'], save = "")
        errors = eo_analysis.error_analysis(postprocessed['test'])
        errors = errors[['Group', 'Method', 'cGFPR', 'cGFNR', 'oGFPR', 'oGFNR']]

        test_df = postprocessed['test']
        roc = FairnessMetricVisualizer(metric = 'roc', parameters = config)
        _roc = roc.visualize_metric(test_df, save = "")

        precision_recall = FairnessMetricVisualizer(metric = 'precision_recall', parameters = config)
        _pr = precision_recall.visualize_metric(test_df, save = "")

        calibration = FairnessMetricVisualizer(metric = 'calibration', parameters = config)
        _calibration = calibration.visualize_metric(test_df, save = "")

        return _eo, _roc, _pr, _calibration


    with ThreadPoolExecutor(max_workers = 8) as executor:
        
        _eo, _roc, _pr, _calibration = [], [], [], []
        for result in executor.map(synthetic_experiment, range(num_seeds)):
            eo, roc, pr, calibration = result
            _eo.append(eo)

        with open('./eo.p', 'wb') as f:
            pickle.dump(_eo, f)
            
        #     __seed = defaultdict(list)
        #     for i, components in enumerate(pr):

        #         if i in [0, 1]:
        #             continue
        #         evaluation = ["Observational", "Control", "Doubly-robust", "True Counterfactual"][i - 2]
        #         for component in components:
        #             recall, precision, _, method = component
        #             __seed[(evaluation, method, 'independent')].append(recall)
        #             __seed[(evaluation, method, 'dependent')].append(precision)
        #     _seed['pr'] = __seed

        # print(_seed['pr'][('True Counterfactual', 'Observational', 'independent')])

            


