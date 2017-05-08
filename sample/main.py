# the usual include statements

from pyspark import SQLContext, SparkContext
from pyspark.sql.types import StructField, StructType, IntegerType

from sample.DataIO import DataIO
from sample.CreateFeatures import AssembleKmeans
import argparse
import os
import sys

PARQUET_PATH = "/home/svanhmic/workspace/data/DABAI/sparkdata/parquet"

if __name__ == '__main__':

    if os.path.exists('jobs.zip'):
        sys.path.insert(0, 'jobs.zip')
    else:
        sys.path.insert(0, './jobs')

    sc = SparkContext("local[*]", "cleaning workflow")
    sqlContext = SQLContext(sc)

    #data_imports = DataIO(sc, PARQUET_PATH + "//featureDataCvr.parquet", PARQUET_PATH + "/companyCvrData")

    #data_imports.get_latest_company(["cvrNummer"], ["periode_gyldigFra"]).show()
    #data_imports.mergeCompanyFeatureData().show()
    params = AssembleKmeans()
    params.select_parameters()

