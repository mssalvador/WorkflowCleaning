# the usual include statements

from pyspark import SparkContext
from pyspark import SQLContext

#from sample.DataIO import DataIO
#from sample.CreateParameters import CreateParameters
#from sample.ExecuteWorkflow import ExecuteWorkflow
#from shared.create_dummy_data import *

import argparse
import os
import sys

import getpass

from scipy.fftpack.pseudo_diffs import sc_diff

user = getpass.getuser()

if user == "sidsel":
    PARQUET_PATH = "/home/" + user + "/workspace/sparkdata/parquet/"

elif user == "svanhmic":
    PARQUET_PATH = "/home/" + user + "/workspace/data/DABAI/sparkdata/parquet/"

TEST_DICT = {'featuresCol': 'scaled_features',
 'k': 2,
 'maxIter': 100,
 'prediction': 'prediction',
 'probability': 'probability',
 'seed': 0,
 'tol': 0.001}


if __name__ == '__main__':

    if os.path.exists('shared.zip'):
        sys.path.insert(0, 'shared.zip')
    else:
        sys.path.insert(0, './shared')

    if os.path.exists('sample.zip'):
        sys.path.insert(0,'sample.zip')
    else:
        sys.path.insert(0, './sample')

    #from shared.create_dummy_data import create_dummy_data
    #from shared.ComputeDataFrameSize import compute_size_of_dataframe

    #n = int(sys.argv[1])
    #if not sys.argv[1]:
    #    n = 100000

    #help(create_dummy_data)
    #df = create_dummy_data(number_of_samples=n,
    #                       feature_names=["x", "y", "z"],
    #                       label_names=["label"],
    #                       outlier_factor=10,
    #                       outlier_number=0.1)
    #df.write.parquet("/user/micsas/data/parquet/"+str(n)+"_samples.parquet", mode='overwrite')
    #compute_size_of_dataframe(df)
    sc = SparkContext().getOrCreate()
    sql_ctx = SQLContext.getOrCreate(sc)
    import numpy as np
    from shared.ComputeDistances import compute_distance
    from pyspark.ml.linalg import VectorUDT, Vectors
    from cleaning.ShowCleaning import ShowResults
    import pandas as pd


    df = sql_ctx.read.parquet('/home/svanhmic/workspace/data/DABAI/sparkdata/parquet/merged_df_parquet')

    shows = ShowResults(TEST_DICT, ['x', 'y', 'z'], ['label'])
    shows.compute_shift(df).show()
