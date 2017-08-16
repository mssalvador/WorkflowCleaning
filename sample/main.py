# the usual include statements

from pyspark import SparkContext

#from sample.DataIO import DataIO
#from sample.CreateParameters import CreateParameters
#from sample.ExecuteWorkflow import ExecuteWorkflow
from shared.create_dummy_data import *

import argparse
import os
import sys

import getpass

user = getpass.getuser()

if user == "sidsel":
    PARQUET_PATH = "/home/" + user + "/workspace/sparkdata/parquet/"

elif user == "svanhmic":
    PARQUET_PATH = "/home/" + user + "/workspace/data/DABAI/sparkdata/parquet/"

TEST_DICT = {'features': ('AarsVaerk_1','AarsVaerk_2','AarsVaerk_3'),
             'initialstep': 10,
             'standardize': False,
             'clusters': 50,
             'model': 'KMeans',
             'initialmode': 'random',
             'prediction': 'Prediction',
             'iterations': 20
             }


if __name__ == '__main__':

    if os.path.exists('shared.zip'):
        sys.path.insert(0, 'shared.zip')
    else:
        sys.path.insert(0, './shared')

    if os.path.exists('sample.zip'):
        sys.path.insert(0,'sample.zip')
    else:
        sys.path.insert(0, './sample')

    from shared.create_dummy_data import create_dummy_data
    from shared.ComputeDataFrameSize import compute_size_of_dataframe

    n = int(sys.argv[1])
    if not sys.argv[1]:
        n = 100000

    #help(create_dummy_data)
    df = create_dummy_data(number_of_samples=n,
                           feature_names=["x", "y", "z"],
                           label_names=["label"],
                           outlier_factor=10,
                           outlier_number=0.1)
    compute_size_of_dataframe(df)