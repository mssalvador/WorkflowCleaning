import numpy as np
import pyspark
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql import functions as F
from shared import ConvertAllToVecToMl
from shared.WorkflowLogger import logger_info_decorator


# def create_complete_graph(data_frame, id_col=None, points=None,
#                           sigma=0.7, standardize=True):
#     """
#     Does a cross-product on the dataframe. And makes sure that the "points" are kept as a vector
#     points: column names that should be the feature vector
#     """
#
#     assert isinstance(points, list), 'this should be a list, not {}'.format(type(points))
#     new_cols = list(set(data_frame.columns) - set(points))
#
#
#     feature_gen = VectorAssembler(inputCols=points, outputCol='features')
#     vector_converter = ConvertAllToVecToMl.ConvertAllToVecToMl(
#         inputCol=feature_gen.getOutputCol(), outputCol='conv_feat')
#     standardizer = StandardScaler(withMean=False, withStd=False,
#         inputCol=vector_converter.getOutputCol(), outputCol='standard_feat')
#
#     if standardize:
#         standardizer.setWithMean(True)
#         standardizer.setWithStd(True)
#
#     pipeline = Pipeline(stages=[feature_gen, vector_converter, standardizer])
#
#     model = pipeline.fit(data_frame)
#     df_cleaned = (model.transform(data_frame)
#         .select(new_cols + [F.col(standardizer.getOutputCol()).alias('features')]))
#
#     a_names = [F.col(name).alias('a_'+name) for name in df_cleaned.columns]
#     b_names = [F.col(name).alias('b_'+name) for name in df_cleaned.columns]
#
#     compute_distance_squared = F.udf(
#         lambda x, y: _compute_weights(x, y, sigma), T.DoubleType())
#
#     distance = compute_distance_squared(F.col('a_features'), F.col('b_features'))
#     return (df_cleaned.select(*a_names).join(df_cleaned.select(*b_names))
#         .withColumn('weights_ab', distance))

@logger_info_decorator
def _compute_weights(vec_x, vec_y, sigma):
    if isinstance(vec_y,SparseVector) | isinstance(vec_x, SparseVector):
        x_d = vec_x.toArray()
        y_d = vec_y.toArray()
        return float(np.exp(-(np.linalg.norm(x_d-y_d, ord=2)**2)/sigma**2))
    else:
        return float(np.exp(-np.linalg.norm(vec_x - vec_y, ord=2)**2 / sigma**2))


@logger_info_decorator
def create_complete_graph(data_frame, feature_columns, id_column='id',
                          label_column='label', standardize=True, sigma=0.7):
    feature_gen = VectorAssembler(inputCols=feature_columns, outputCol='features')
    vector_converter = ConvertAllToVecToMl.ConvertAllToVecToMl(
        inputCol=feature_gen.getOutputCol(), outputCol='conv_feat')
    standardizer = StandardScaler(withMean=False, withStd=False,
        inputCol=vector_converter.getOutputCol(), outputCol='standard_feat')

    if standardize:
        standardizer.setWithMean(True)
        standardizer.setWithStd(True)

    pipeline = Pipeline(stages=[feature_gen, vector_converter, standardizer])
    model = pipeline.fit(data_frame)

    rdd = (model.transform(data_frame)
        .select(id_column, label_column, F.col(standardizer.getOutputCol()).alias('features'))
        .rdd
    )
    output_column_types = [
        ('row', T.IntegerType(), False), ('row_label', T.DoubleType(), True),
        ('column', T.IntegerType(), False), ('column_label', T.DoubleType(), True),
        ('weights_ab', T.DoubleType(), False)]
    schema = T.StructType([T.StructField(*i) for i in output_column_types])

    cartesian_rows = pyspark.Row(list(map(lambda x: x[0], output_column_types)))
    second_rdd = rdd.cartesian(rdd)#.filter(lambda x: x[0][0] < x[1][0])
    distance_rdd = second_rdd.map(lambda v: cartesian_rows(
        v[0][id_column], v[0][label_column], v[1][id_column], v[1][label_column],
        _compute_weights(vec_x=v[0]['features'], vec_y=v[1]['features'], sigma=sigma)))
    return distance_rdd.toDF(schema=schema)