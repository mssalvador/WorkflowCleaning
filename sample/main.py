# the usual include statements

from pyspark import SparkContext
import pyspark
from semisupervised import LabelPropagation
import getpass
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

user = getpass.getuser()
if user == "sidsel":
    PARQUET_PATH = "/home/" + user + "/workspace/sparkdata/parquet/"

elif user == "svanhmic":
    PARQUET_PATH = "/home/" + user + "/workspace/data/DABAI/sparkdata/parquet/"

if __name__ == '__main__':


    sc = SparkContext.getOrCreate()
    spark_session = pyspark.sql.SparkSession(sc)
    spark_session.conf.set("spark.sql.crossJoin.enabled", "true")


    data_1 = {'label': [0.0] + 3 * [None],
              'x': np.random.normal(size=4),
              'y': np.random.normal(size=4)
              }
    pdf_1 = pd.DataFrame(data_1, columns=['label', 'x', 'y'])

    data_2 = data_1 = {'label': [1.0] + 3 * [None],
                       'x': np.random.normal(4, .5, size=4),
                       'y': np.random.normal(4, .5, size=4)
                       }
    pdf_2 = pd.DataFrame(data_2, columns=['label', 'x', 'y'])
    pdf = pd.concat([pdf_1, pdf_2], ignore_index=True).reset_index()
    df_input = spark_session.createDataFrame(pdf)
    df_input.show()



    summed_transition = LabelPropagation.label_propagation(
        sc= sc, data_frame= df_input,
        label_col= 'label', id_col= 'index',
        feature_cols= ['x','y'], k= 2, tol=0.000001,
        max_iters=25)

    summed_transition.show()