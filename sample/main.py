# the usual include statements

from pyspark import SparkContext
from pyspark.sql import SparkSession
from semisupervised import LabelPropagation
from pyspark.ml.linalg import DenseVector
from cleaning.ShowCleaning import ShowResults
from pyspark.sql import functions as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import getpass

user = getpass.getuser()
if user == "sidsel":
    PARQUET_PATH = "/home/" + user + "/workspace/sparkdata/parquet/"

elif user == "svanhmic":
    PARQUET_PATH = "/home/" + user + "/workspace/data/DABAI/sparkdata/parquet/"

# if __name__ == '__main__':
#
#     sc = SparkContext.getOrCreate()
#     spark_session = SparkSession(sc)
#
#     ShowResults(sc,
#                 {'predictionCol': [1, 1], 'k': 2},
#                 ['feat1', 'feat2'], ['lab1', 'lab2']
#                 )
#     mini_pdf = pd.DataFrame(
#         {'predictionCol': [0, 0, 0, 0, 0, 0, 1], 'distance': [0.5, 1.5, 0.5, 0.1, 0.01, 6.0, 20.0]},
#         columns=['predictionCol', 'distance']
#     )
#
#     dataframe = spark_session.createDataFrame(mini_pdf)
#
#     preped_df = ShowResults.add_outliers(dataframe)
#     ShowResults.compute_summary(preped_df).show()

if __name__ == '__main__':

    helix = PARQUET_PATH+'double_helix.parquet'

    sc = SparkContext.getOrCreate()
    spark_session = SparkSession(sc)
    spark_session.conf.set("spark.sql.crossJoin.enabled", "true")

    id_col = np.array([0, 1, 2, 3])
    # np.random.shuffle(id_col)
    data = {
        'label': [0.0, 1.0] + 2 * [None],
        'x': np.array([0., 0.9, 0.1, 0.85]),
        'y': np.array([0., 0.9, 0.1, 0.85]),
        'z': np.array([0., 0.9, 0.1, 0.85]),
    }
    pdf = pd.DataFrame(data, columns=['label', 'x', 'y', 'z'])
    pdf['id'] = id_col

    #df_input = spark_session.createDataFrame(pdf)
    df_input = spark_session.read.parquet(helix)
    summed_transition = LabelPropagation.label_propagation(
        sc= sc, data_frame= df_input,
        label_col= 'label', id_col= 'id',
        feature_cols= ['x','y','z'], k= 2, tol=0.000001,
        max_iters=25, sigma=0.43)

    #summed_transition.show(truncate=False)
    combined_hack_df = df_input.select(['x','y','z','id']).alias('a').join(
        summed_transition.alias('b'), on= F.col('a.id') == F.col('b.row'),
        how= 'inner').drop('b.row')
    combined_hack_df.drop('row_trans').show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pandas_transition = combined_hack_df.toPandas()
    for i in range(2):
        ax.scatter(
            pandas_transition[pandas_transition['label'] == i]['x'],
            pandas_transition[pandas_transition['label'] == i]['y'],
            pandas_transition[pandas_transition['label'] == i]['z'])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Our test dataset a double helix')
    plt.show()

