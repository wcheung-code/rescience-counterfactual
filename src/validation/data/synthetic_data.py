import os
import glob
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


DATA_DIRECTORY = './data/validation/01-data_generation'

interpreter = {
    'python': 'python3',
    'r': 'Rscript'
}

suffix = {
    'python': 'py',
    'r': 'R'
}

## Generate synthetic data files for multiple seeds
base, treatment = {}, {}
for language in ['python', 'r']:

    base_rates = defaultdict(list)
    treatment_rates = defaultdict(list)

    dir_name = f'{DATA_DIRECTORY}/{language}'
    if os.path.isdir(dir_name):
        if not glob.glob(os.path.join(dir_name, '*.csv')):
            os.system(f"{interpreter[language]} ./src/validation/data/{language}/make_dataset.{suffix[language]} --num_seeds=30")
    else:
        print(f"directory {dir_name} doesn't exist")

    for filepath in glob.glob(os.path.join(dir_name, '*.csv')):
        if language == 'python':
            df = pd.read_csv(filepath)
            for column in ['Y', 'Y_1', 'Y_0']:
                base_rates[''.join(column.split('_'))].append(df[column].mean())
        else:
            df = pd.read_csv(filepath)
            for column in ['y', 'y1', 'y0']:
                base_rates[column.upper()].append(df[column].mean())
        for column in ['all', 'A = 0', 'A = 1']:
            if "=" not in column: 
                treatment_rates[column].append(df.treat_num.mean())
            else:
                value = int(column.split(' = ')[-1])
                treatment_rates[column].append(df[df.A == value].treat_num.mean())

    base[language] = base_rates
    treatment[language] = treatment_rates

print(base)
print(treatment)





