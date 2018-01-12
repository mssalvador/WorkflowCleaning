from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from pyspark.mllib import linalg
from pyspark.mllib.linalg import distributed
import numpy as np


def triangle_mat_summation(mat_element):
    if mat_element.j == mat_element.i:
        return (mat_element.i, mat_element.value),
    else:
        return (mat_element.i, mat_element.value), (mat_element.j, mat_element.value)


def generate_label_matrix(df, label_col='label', id_col='id', k=None):
    null_nan_check = (~F.isnan(F.col(label_col)) & (~F.isnull(F.col(label_col))))
    y_known = (df.filter(null_nan_check).select(id_col, label_col).cache())
    y_unknown = df.filter(~null_nan_check).select(id_col).cache()
    if k == None:
        y_range = y_known.select(label_col).distinct().count()
    else:
        y_range = int(k)

    y_known_rdd = y_known.rdd.map(
        lambda x: distributed.MatrixEntry(
            i=int(x[id_col]), j=int(x[label_col]), value=1.0)
    )
    y_unknown_rdd = y_unknown.rdd.flatMap(
        lambda x: [distributed.MatrixEntry(
            i=int(x[id_col]), j=idx, value=.50) for idx in range(int(y_range))]
    )
    y_matrix = distributed.CoordinateMatrix(
        entries=y_unknown_rdd.union(y_known_rdd), numRows=df.count(), numCols=y_range)
    return y_known_rdd, y_matrix


def merge_data_with_label(sc, org_data_frame, coordinate_label_rdd, id_col='id'):
    spark = SparkSession(sc)
    indexed_label_rdd = (coordinate_label_rdd
        .toIndexedRowMatrix().rows
        .map(lambda x: (x.index, x.vector))) # get rdd with row_index and columns

    schema = (T.StructType().add(field=id_col, data_type=T.IntegerType())
        .add('probabilities', data_type=linalg.VectorUDT()))

    final_label_data_frame = spark.createDataFrame(data=indexed_label_rdd, schema=schema)
    merged_data_frame = org_data_frame.join(final_label_data_frame, on=id_col, how='left')
    return merged_data_frame


def evaluate_label_based_on_eval(sc, data_frame, label_col='label', **kwargs):
    priors = kwargs.get('priors', None)
    assert 'probabilities' in data_frame.columns, 'Column not in dataframe!'
    if ~isinstance(priors, list):
        if kwargs.get('eval_type', None) == 'max':
            priors = (data_frame
                .filter((~F.isnull(label_col) | ~F.isnan(label_col)))
                .groupBy(label_col).count().rdd.map(lambda x: [i for i in x])
                .collectAsMap())
        else: #
            max_label = F.udf(lambda x: float(np.argmax(x.toArray())), T.DoubleType())
            return data_frame.withColumn(colName='new_'+label_col, col=max_label('probabilities'))

    broad_cast_prior = sc.broadcast(np.array(priors))
    udf_dot = lambda x: np.multiply(x.toArray(), broad_cast_prior.value).tolist()
    udf_type = T.ArrayType(T.DoubleType(), containsNull=False)
    return data_frame.withColumn(
        colName='probabilities',
        col=F.udf(lambda x: udf_dot, udf_type)('probabilities'))
