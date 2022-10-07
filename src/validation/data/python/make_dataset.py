# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

import numpy as np
import pandas as pd
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

from SyntheticData import SyntheticData

if __name__ == '__main__':

    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    # dotenv_path = os.path.join(project_dir, '.env')
    # dotenv.load_dotenv(dotenv_path)
    DATA_DIRECTORY = './data/validation/01-data_generation/python/'

    c, k = 0.1, 1.6
    num_points = 100000

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", help="number of random seeds", default=30)
    args = parser.parse_args()
    num_seeds = int(args.num_seeds)

    def generate_data(seed, c = c, k = k, num_points = num_points):
        synthetic_data = SyntheticData(
                        treatment_effect = c, 
                        treatment_assignment_bias = k,
                        seed = seed)
        df, _ = synthetic_data.generate(num_points = num_points)
        df.to_csv(os.path.join(DATA_DIRECTORY, f'seed_{str(seed).zfill(3)}.csv'), encoding='utf-8', index=False)

    with ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(generate_data, range(num_seeds))


