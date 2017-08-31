# the usual include statements

from pyspark import SparkContext
from pyspark import SQLContext

from shared import create_dummy_data
from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow
from shared import Plot2DGraphs
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import argparse
import os
import sys

import getpass

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

    dim = 2
    k = 10
    samples = 10000

    for i in [10000000]:
        means = create_dummy_data.create_means(dim, k, 10)  # [[0, 0, 0], [3, 3, 3], [-3, 3, -3], [5, -5, 5]]
        stds = create_dummy_data.create_stds(dim, k)  # [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        n_samples = create_dummy_data.create_partition_samples(i, k)  # [1000, 10000, 4000, 50]
        print(n_samples)
        df = create_dummy_data.create_normal_cluster_data_spark(dim, n_samples, means, stds)
        #df.show(100)
        df.write.parquet(PARQUET_PATH+'normal_cluster_n_'+str(i)+'.parquet', mode='overwrite')

    # test_params_1 = {'tol': 0.001, 'k': 8, 'maxIter': 10, 'algorithm': 'GaussianMixture', 'seed': 1080866016001745000}

    # execution_model = ExecuteWorkflow(dict_params=test_params_1
    #                                   , cols_features=['a', 'b']
    #                                   , cols_labels=['id', 'k', 'dimension'])

    #pipeline = execution_model.execute_pipeline(df)
    #transformed = execution_model.apply_model(pipeline, df)
    #transformed.show()

    #Plot2DGraphs.plot_gaussians(transformed, ['a', 'b'])
    #plt.show()