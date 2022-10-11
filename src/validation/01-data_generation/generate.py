import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm, ks_2samp
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

DATA_DIRECTORY = './data/validation/01-data_generation'
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

# expected_values = {
#     'Y': 0.17,
#     'Y^1': 0.04,
#     'Y^0': 0.4,
#     'T \mid A = 0': 0.4,
#     'T \mid A = 1': 0.71,
#     'T': 0.55
# }

def generate_means(filename):
    df = pd.read_csv(filename)
    language = str(Path(filename).parent.absolute()).split('/')[-1]
    columns = ['y', 'y1', 'y0'] if language == 'r' else ['Y', 'Y_1', 'Y_0']
    results = df[columns].mean()
    for a, _df in df.groupby('A'):
        results.at[f'T \\mid A = {a}'] = _df.treat_num.mean()
    results.at['T'] = df.treat_num.mean()
    if language == 'r':
        return results.rename({'y': 'Y', 'y1' : 'Y^1', 'y0': 'Y^0'})
    else:
        return results.rename({'Y': 'Y', 'Y_1' : 'Y^1', 'Y_0': 'Y^0'})
## Generate synthetic data files for multiple seeds

if __name__ == '__main__':

    # TODO: Uncomment for Docker container
#    os.makedirs(f"{FIGURE_DIRECTORY}/01-data_generation", exist_ok = True)

    rates, extracts = {}, []

    for language in ['python', 'r']:
        dir_name = f'{DATA_DIRECTORY}/{language}'
        if os.path.isdir(dir_name):
            if not glob.glob(os.path.join(dir_name, '*.csv')):
                os.system(f"{interpreter[language]} ./src/validation/01-data_generation/{language}/make_dataset.{suffix[language]} --num_seeds=150")
        else:
            print(f"directory {dir_name} doesn't exist")
    
    for language in ['python', 'r']:
        dir_name = f'{DATA_DIRECTORY}/{language}'
        with ThreadPoolExecutor(max_workers = 8) as executor:
            for response in executor.map(generate_means, glob.glob(os.path.join(dir_name, '*.csv'))):
                extracts.append(response)
        
        rates[language] = pd.DataFrame(extracts)

    print('Shapiro-Wilk Test Results:')
    print('---------------------------')
    print('Python:')
    for col in rates['python'].columns:
        print(f"P-value for column {col}: {shapiro(sorted(rates['python'][col]))}")
    print('---------------------------')
    print('R:')
    for col in rates['r'].columns:
        print(f"P-value for column {col}: {shapiro(sorted(rates['r'][col]))}")
    print('---------------------------')
    print('Kolmogorov-Smirnov Test Results:')
    print('---------------------------')
    for col in rates['python'].columns:
        print(f"P-value for column {col}: {ks_2samp(sorted(rates['python'][col]), sorted(rates['r'][col]))}")
    print('---------------------------')

    for i, col in enumerate(rates['python'].columns):
        python_data = rates['python'][col]
        r_data = rates['r'][col]

        mu_py, std_py = norm.fit(python_data)
        mu_r, std_r = norm.fit(r_data) 

        plt.figure(i)

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

        title = f"Synthetic Data Generation Replication Results (Feature: ${col}$)"
        plt.title(title)
        plt.legend(loc='upper right')
        # plt.savefig(f"./reports/figures/validation/fig_synth_data_generation_{i}.png")
        # TODO: Uncomment for Docker container
        plt.savefig(f"{FIGURE_DIRECTORY}/01-data_generation/fig_synth_data_generation_{i}.png")