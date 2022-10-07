import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm, ks_2samp
from collections import defaultdict
import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


DATA_DIRECTORY = './data/validation/01-data_generation'

interpreter = {
    'python': 'python3',
    'r': 'Rscript'
}

suffix = {
    'python': 'py',
    'r': 'R'
}

def generate_means(filename):
    df = pd.read_csv(filename)
    language = str(Path(filename).parent.absolute()).split('/')[-1]
    columns = ['y', 'y1', 'y0'] if language == 'r' else ['Y', 'Y_1', 'Y_0']
    results = df[columns].mean()
    for a, _df in df.groupby('A'):
        results.at[f'T(A_{a})'] = _df.treat_num.mean()
    results.at['T'] = df.treat_num.mean()
    return results.rename({'y': 'Y', 'y1' : 'Y_1', 'y0': 'Y_0'})

## Generate synthetic data files for multiple seeds

if __name__ == '__main__':

    rates = {}
    for language in ['python', 'r']:
        extracts, dir_name = [], f'{DATA_DIRECTORY}/{language}'
        if os.path.isdir(dir_name):
            if not glob.glob(os.path.join(dir_name, '*.csv')):
                os.system(f"{interpreter[language]} ./src/validation/data/{language}/make_dataset.{suffix[language]} --num_seeds=150")
        else:
            print(f"directory {dir_name} doesn't exist")

        with ThreadPoolExecutor(max_workers = 8) as executor:
            for response in executor.map(generate_means, glob.glob(os.path.join(dir_name, '*.csv'))):
                extracts.append(response)
        
        rates[language] = pd.DataFrame(extracts)

    print('Shapiro-Wilk Test Results:')
    print('---------------------------')
    print('Python:')
    for col in rates['python'].columns:
        print(f"P-value for column {col}: {shapiro(rates['python'][col])}")
    print('---------------------------')
    print('R:')
    for col in rates['r'].columns:
        print(f"P-value for column {col}: {shapiro(rates['r'][col])}")
    print('---------------------------')
    print('Kolmogorov-Smirnov Test Results:')
    print('---------------------------')
    for col in rates['python'].columns:
        print(f"P-value for column {col}: {ks_2samp(rates['python'][col], rates['r'][col])}")
    print('---------------------------')

    for i, col in enumerate(rates['python'].columns):
        python_data = rates['python'][col]
        r_data = rates['r'][col]

        mu_py, std_py = norm.fit(python_data)
        mu_r, std_r = norm.fit(r_data) 

        plt.figure(i)

        plt.hist(python_data, bins=25, density=True, alpha=0.3, label = 'python')
        plt.hist(r_data, bins=25, density=True, alpha=0.3, label = 'R')

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p_py = norm.pdf(x, mu_py, std_py)
        p_r = norm.pdf(x, mu_r, std_r)

        plt.plot(x, p_py, linewidth=2, alpha=0.7, c='#1f77b4')
        plt.plot(x, p_r, linewidth=2, alpha=0.7, c='#ff7f0e')
        min_x = min(min(python_data), min(r_data))
        max_x = max(max(python_data), max(r_data))
        plt.xlim([min_x, max_x])
        plt.yticks([])

        title = f"Synthetic Data Generation Replication Results (Feature: ${col}$)"
        plt.title(title)
        plt.legend(loc='upper right')
        plt.savefig(f"./reports/figures/validation/01-data_generation/fig_synth_data_generation_{col}.png")


  
# # Generate some data for this 
# # demonstration.
# python_data = rates['python']['Y']
# r_data = rates['r']['Y']
  
# # Fit a normal distribution to
# # the data:
# # mean and standard deviation
# mu_py, std_py = norm.fit(python_data)
# mu_r, std_r = norm.fit(r_data) 
  
# # Plot the histogram.
# plt.hist(python_data, bins=30, density=True, alpha=0.3, label = 'python')
# plt.hist(r_data, bins=30, density=True, alpha=0.3, label = 'R')
  
# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p_py = norm.pdf(x, mu_py, std_py)
# p_r = norm.pdf(x, mu_r, std_r)


# plt.plot(x, p_py, linewidth=2, alpha=0.7, c='#1f77b4')
# plt.plot(x, p_r, linewidth=2, alpha=0.7, c='#ff7f0e')
# min_x = min(min(python_data), min(r_data))
# max_x = max(max(python_data), max(r_data))
# plt.xlim([min_x, max_x])
# plt.yticks([])

# title = f"Synthetic data generation replication results (Feature: {})"
# plt.title(title)

# # bins = np.linspace(0.16, 0.17, 100)
# # plt.hist(base['python']['Y'], bins, alpha=0.3, label='python')
# # mu, std = norm.fit()

# # plt.hist(base['r']['Y'], bins, alpha=0.3, label='r')
# plt.legend(loc='upper right')
# plt.savefig('./reports/figures/validation/foo.png')



