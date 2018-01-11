from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import distributed


def triangle_mat_summation(mat_element):
    if mat_element.j == mat_element.i:
        return (mat_element.i, mat_element.value),
    else:
        return (mat_element.i, mat_element.value), (mat_element.j, mat_element.value)


def generate_label_matrix(df, label_col = 'label', id_col = 'id'):
    y_known = df.filter(~F.isnan(F.col(label_col))).select(id_col, label_col).cache()
    y_unknown = df.filter(F.isnan(F.col(label_col))).select(id_col).cache()
    y_max = y_known.groupby().max(label_col).collect()[0][0]

    y_known_rdd = y_known.rdd.map(
        lambda x: distributed.MatrixEntry(
            i=x[id_col], j=x[label_col], value=1.0)
    )
    y_unknown_rdd = y_unknown.rdd.flatMap(
        lambda x: [distributed.MatrixEntry(
            i=x[id_col], j=idx, value=.50) for idx in range(int(y_max + 1))]
    )
    y_matrix = distributed.CoordinateMatrix(
        entries=y_unknown_rdd.union(y_known_rdd), numRows=df.count(), numCols=y_max + 1)
    return y_matrix


def merge_data_with_label(sc, org_data_frame, label_rdd, id_col='id'):
    spark = SparkSession(sc)
    inner_struct = T.StructType().add(
        field='probability', data_type=T.DoubleType()).add(
        field='1-probability', data_type=T.DoubleType())
    schema = T.StructType().add(field=id_col, data_type=T.IntegerType()).add('probabilities', data_type=inner_struct)

    final_label_data_frame = spark.createDataFrame(data=label_rdd, schema=schema)
    merged_data_frame = org_data_frame.join(final_label_data_frame, on=id_col, how='left')
    return merged_data_frame
