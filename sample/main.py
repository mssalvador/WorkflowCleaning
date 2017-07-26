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

    if os.path.exists('jobs.zip'):
        sys.path.insert(0, 'jobs.zip')
    else:
        sys.path.insert(0, './jobs')

    # sc = SparkContext.getOrCreate()
    # dd = DummyData(sc)
    #
    # # examples on splitting
    # ddx = dd.df[dd.df["x"] > 0.5]
    # ddx.show(5)
    #
    # # slicing a data frame is as follows
    #
    # ddy = dd.df["label", "x"]
    # ddy.show(5)
    # labels = ['header_1', 'header_2', 'header_3']
    # features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    # df_outliers = create_dummy_data(number_of_samples=1000000,
    #                                 labels=labels,
    #                                 features=features,
    #                                 outlier_factor=50,
    #                                 outlier_number=0.5
    #                                 )
    # df_outliers.show()
    # print(df_outliers.count())

    # df = make_outliers(df_outliers, 0.5, 100)
    # features_of_list = map(lambda f: '(' + str(f) + ' > 1)', [i[0] for i in df_outliers.dtypes if i[1] == 'double'])
    # print('Number of outliers = ' + str(df_outliers.where(' or '.join(features_of_list)).count()))
    # #df.show()
    # df_outliers.write.parquet(PARQUET_PATH+'outlier_df.parquet', mode='overwrite')

    from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification

    ex = ExecuteWorkflowClassification()

    ex.create_custom_pipeline()