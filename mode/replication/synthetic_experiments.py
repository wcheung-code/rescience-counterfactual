import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.SyntheticData import SyntheticData
from src.SupervisedLearningModel import SupervisedLearningModel
from src.ReweighingAnalysisVisualizer import ReweighingAnalysisVisualizer
from src.MetricFrameGenerator import MetricFrameGenerator
from src.EqualizedOddsPostProcesser import EqualizedOddsPostProcesser
from src.EqualizedOddsPostProcessingAnalysis import EqualizedOddsPostProcessingAnalysis
from src.FairnessMetricVisualizer import FairnessMetricVisualizer

if __name__ == '__main__':

    FIGURE_DIRECTORY = "./replication"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="random integer seed", default=0)
    args = parser.parse_args()
    seed = int(args.seed)

    c, k = 0.1, 1.6
    num_points = 100000

    model_list = ['propensity', 'observational', 'counterfactual']

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

    reweighing_analysis = ReweighingAnalysisVisualizer(config)
    reweighing_analysis.visualize_base_rates(df, save = f"{FIGURE_DIRECTORY}/reweighing/fig_seed_{str(seed).zfill(3)}.png")

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
    _eo = eo_analysis.visualize_roc(postprocessed['test'], save = f"{FIGURE_DIRECTORY}/post_processed/fig_roc_seed_{str(seed).zfill(3)}.png")
    errors = eo_analysis.error_analysis(postprocessed['test'])
    errors = errors[['Group', 'Method', 'cGFPR', 'cGFNR', 'oGFPR', 'oGFNR']]
    with open( f"{FIGURE_DIRECTORY}/post_processed/fig_roc_seed_{str(seed).zfill(3)}.tex", 'w') as f:
        f.write(errors.to_latex(column_format='llrrrr', index=False))

    test_df = postprocessed['test']
    roc = FairnessMetricVisualizer(metric = 'roc', parameters = config)
    _roc = roc.visualize_metric(test_df, save = f"{FIGURE_DIRECTORY}/roc/fig_seed_{str(seed).zfill(3)}.png")

    precision_recall = FairnessMetricVisualizer(metric = 'precision_recall', parameters = config)
    _pr = precision_recall.visualize_metric(test_df, save = f"{FIGURE_DIRECTORY}/precision_recall/fig_seed_{str(seed).zfill(3)}.png")

    calibration = FairnessMetricVisualizer(metric = 'calibration', parameters = config)
    _calibration = calibration.visualize_metric(test_df, save = f"{FIGURE_DIRECTORY}/calibration/fig_seed_{str(seed).zfill(3)}.png")