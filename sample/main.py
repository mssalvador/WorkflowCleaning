# the usual include statements

from pyspark import SparkContext

#from sample.DataIO import DataIO
#from sample.CreateParameters import CreateParameters
#from sample.ExecuteWorkflow import ExecuteWorkflow
from shared.create_dummy_data import DummyData

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

    if os.path.exists('jobs.zip'):
        sys.path.insert(0, 'jobs.zip')
    else:
        sys.path.insert(0, './jobs')

    sc = SparkContext.getOrCreate()
    dd = DummyData(sc)

    # examples on splitting
    ddx = dd.df[dd.df["x"] > 0.5]
    ddx.show(5)

    # slicing a data frame is as follows

    ddy = dd.df["label", "x"]
    ddy.show(5)