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

import pandas as pd
import json
import os

from SyntheticData import SyntheticData

if __name__ == '__main__':

    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    # dotenv_path = os.path.join(project_dir, '.env')
    # dotenv.load_dotenv(dotenv_path)
    DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY")
    print(DATA_DIRECTORY)

    c, k = 0.1, 1.6
    num_points = 100000

    synthetic_data = SyntheticData(
                               treatment_effect = c, 
                               treatment_assignment_bias = k,
                               seed = 1)
    df, config = synthetic_data.generate(num_points = num_points)


    df.to_csv(os.path.join(DATA_DIRECTORY, 'raw', 'synthetic_data.csv'), encoding='utf-8', index=False)
    with open(os.path.join(DATA_DIRECTORY, 'config', 'synthetic_config.json'), "w") as f:
        json.dump(config, f)
