import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm, ks_2samp, anderson_ksamp
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import rc

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex = True)
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

DATA_DIRECTORY = './data/validation/02-model_training'
# TODO: Uncomment for Docker container
#FIGURE_DIRECTORY = './validation'

interpreter = {
    'python': 'python3',
    'r': 'Rscript'
}

suffix = {
    'python': 'py',
    'r': 'R'
}

def extract_coefs(filename):
    df = pd.read_csv(filename, index_col = 0)
    return df.to_numpy()

## Generate synthetic data files for multiple seeds

if __name__ == '__main__':

    # TODO: Uncomment for Docker container
    #os.makedirs(f"{FIGURE_DIRECTORY}/02-model_training", exist_ok = True)

    tensor = {}

    for language in ['python', 'r']:
        dir_name = f'{DATA_DIRECTORY}/{language}'
        if os.path.isdir(dir_name):
            if not glob.glob(os.path.join(dir_name, '*.csv')):
                os.system(f"{interpreter[language]} ./src/validation/02-model_training/{language}/fit_model.{suffix[language]} --num_seeds=150")
        else:
            print(f"directory {dir_name} doesn't exist")
    
    for language in ['python', 'r']:
        coefficients = np.empty((3, 3, 150)) 
        dir_name = f'{DATA_DIRECTORY}/{language}'
        with ThreadPoolExecutor(max_workers = 8) as executor:
            for i, layer in enumerate(executor.map(extract_coefs, glob.glob(os.path.join(dir_name, '*.csv')))):
                coefficients[:, :, i] = layer
        tensor[language] = coefficients

    features, models = ['Intercept', 'Z', 'A'], ['propensity', 'observational', 'counterfactual']

    for i, feature in enumerate(features):
        for j, model in enumerate(models):
            print(feature, model)

            print('Shapiro-Wilk Test Results:')
            print('---------------------------')
            print('Python:')
            print(f"P-value for {(feature, model)}: {shapiro(tensor['python'][i, j, :])}")
            print('---------------------------')
            print('R:')
            print(f"P-value for {(feature, model)}: {shapiro(tensor['r'][i, j, :])}")
            print('---------------------------')
            print('Kolmogorov-Smirnov Test Results:')
            print('---------------------------')
            print(f"P-value for {(feature, model)}: {ks_2samp(tensor['python'][i, j, :], tensor['r'][i, j, :])}")
            print('---------------------------')
            print('Anderson-Darling Test Results:')
            print('---------------------------')
            print(f"P-value for {(feature, model)}: {anderson_ksamp([tensor['python'][i, j, :], tensor['r'][i, j, :]])}")
            print('---------------------------')

            python_data = tensor['python'][i, j, :]
            r_data = tensor['r'][i, j, :]

            mu_py, std_py = norm.fit(python_data)
            mu_r, std_r = norm.fit(r_data) 

            plt.figure(10*i + j)

            plt.hist(python_data, bins=30, density=True, alpha=0.3, label = 'python')
            plt.hist(r_data, bins=30, density=True, alpha=0.3, label = 'R')

            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p_py = norm.pdf(x, mu_py, std_py)
            p_r = norm.pdf(x, mu_r, std_r)

            plt.plot(x, p_py, linewidth=2, alpha=0.7, c='#1f77b4')
            plt.plot(x, p_r, linewidth=2, alpha=0.7, c='#ff7f0e')
            #print(col, expected_values[col])
            #plt.axvline(x = expected_values[col], alpha=0.7, c='#000000', label = f"$\\mathbb{{E}}({col})$")
            min_x = min(min(python_data), min(r_data))
            max_x = max(max(python_data), max(r_data))
            plt.xlim([min_x, max_x])
            plt.yticks([])

            title = f"{model.capitalize()} Model Fitting Results (Coefficient: ${feature}$)"
            plt.title(title)
            plt.legend(loc='upper right')
            plt.savefig(f"./reports/figures/validation/fig_synth_data_generation_{10*i + j}.png")
            # TODO: Uncomment for Docker container
    #        plt.savefig(f"{FIGURE_DIRECTORY}/02-model_training/fig_synth_data_generation_{i}.png")