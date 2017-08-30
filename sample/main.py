# the usual include statements

from pyspark import SparkContext
from pyspark import SQLContext

from shared.create_dummy_data import create_normal_cluster_data_spark

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
    means = [[0, 0, 0], [3, 3, 3], [-3, 3, -3], [5, -5, 5]]
    stds = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    n_samples = [1000, 10000, 4000, 50]
    df = create_normal_cluster_data_spark(3, n_samples, means, stds)
    # df.show()
    df.write.parquet(PARQUET_PATH+'normal_cluster_data.parquet', mode='overwrite')
