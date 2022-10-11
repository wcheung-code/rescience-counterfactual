import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import argparse
from concurrent.futures import ThreadPoolExecutor

from RiskAssessmentIndicatorModel import RiskAssessmentIndicatorModel

DATA_DIRECTORY = "./data/validation/02-model_training/python"
RAW_DATA_DIRECTORY = "./data/validation/01-data_generation/python"

config = {
    'treat': {
        'name': 'treat_num'
    },
    'outcome': {
        'name': 'outcome'
    },
    'features': {
        'training': ['Z', 'A'],
        'sensitive': 'A'
    },
    'target': {
        'observational': 'Y',
        'counterfactual': 'Y_0'
    },
    'is_train': False,
    'sample_weight': None
}

if __name__ == '__main__':

    df = pd.read_csv(f"{RAW_DATA_DIRECTORY}/seed_000.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", help="number of random seeds", default=150)
    args = parser.parse_args()
    num_seeds = int(args.num_seeds)

    def fit_models(seed):
        # ## Train classifiers to produce RAI's
        models, model_list = {}, ['propensity', 'observational', 'counterfactual']
        for model_key in model_list:
            params = config.copy()
            clf = LogisticRegression(penalty = 'none')
            rai = RiskAssessmentIndicatorModel(model = clf, name = model_key, seed = seed)
            if model_key == 'propensity':
                params['target'] = params['treat']['name']
            else:
                params['target'] = params['outcome']['name']
            rai.fit(df, params)
            models[model_key] = clf

        coefficients = {k: v.coef_ for k, v in models.items()}
        intercepts = {k: v.intercept_ for k, v in models.items()}
        result = pd.DataFrame(index = ['(Intercept)', 'Z', 'A'])

        for model_key in model_list:
            result[model_key] = np.append(intercepts[model_key], coefficients[model_key])

        result.to_csv(f"{DATA_DIRECTORY}/seed_{str(seed).zfill(3)}.csv")

    with ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(fit_models, range(num_seeds))