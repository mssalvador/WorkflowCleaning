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
    labels = ['header_1', 'header_2', 'header_3']
    features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    df_outliers = create_dummy_data(number_of_samples=30,
                                    labels=labels,
                                    features=features,
                                    #outlier_factor=100,
                                    #outlier_number=5)
                                    )
    #df_outliers.show()
    #print(df_outliers.count())

    df = make_outliers(df_outliers, 0.5, 100)
    # print('Number of outliers = ' + str(df.where(' or '.join(map(lambda f: '(' + f + ' > 1)',
    #                                                              [f[0] for f in df.dtypes if f[1] == 'float']))).count()))
    # df.show()
    # #df_outliers.write.parquet('/home/sidsel/workspace/sparkdata/parquet/outlier_df.parquet')

    from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification
    from pyspark.sql import types as T
    Test = {'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
            'labelCol': 'label',
            'maxIter': (100, 150),
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction'}

    feat = T.StructType([T.StructField("f1",T.FloatType(),True)
                            ,T.StructField("f2",T.FloatType(),True)
                           ,T.StructField("f3",T.FloatType(),True)
                           ,T.StructField("f4",T.FloatType(),True)])

    lab = T.StructType([T.StructField("h1",T.StringType(),True)
                           ,T.StructField("h2",T.StringType(),True)
                           ,T.StructField("h3",T.StringType(),True)])

    exclass = ExecuteWorkflowClassification(Test, True, feat, lab)

    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    exclass.run_cross_val(df, BinaryClassificationEvaluator(), 3)