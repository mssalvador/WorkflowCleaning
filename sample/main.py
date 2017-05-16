# the usual include statements

from pyspark import SQLContext, SparkContext
from pyspark.sql.types import StructField, StructType, IntegerType

from sample.DataIO import DataIO
from sample.CreateFeatures import AssembleKmeans
from sample.ExecuteWorkflow import ExecuteWorkflow

import argparse
import os
import sys

PARQUET_PATH = "/home/svanhmic/workspace/data/DABAI/sparkdata/parquet"

TEST_DICT = {'features': ('AarsVaerk_1',),
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

    sc = SparkContext.getOrCreate()#SparkContext("local[*]", "cleaning workflow")
    #sqlContext = SQLContext(sc)

    data = DataIO(sc,
                  feature_path=PARQUET_PATH + "/featureDataCvr.parquet",
                  company_path=PARQUET_PATH + "/companyCvrData")
    feature_data = data.mergeCompanyFeatureData()

    feature_data.show()

    work_flow = ExecuteWorkflow()
    work_flow.params = TEST_DICT
    work_flow.run(feature_data)

