from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
import math


def subset_dataset_by_label(sc, dataframe, label_col, *args):
    if args:
        return dataframe.filter(F.col(label_col).isin(args))
    else:
        return dataframe


def enlarge_dataset(sc, dataframe, factor):
    return dataframe


def create_nan_labels(sc, dataframe, label_col, fraction=None, **kwargs):
    """
    Generates a column with either a missing factor for each label
    or n missing labels pr. label
    :param sc:
    :param dataframe:
    :param label_col:
    :param args:
    :return:
    """
    spark = SparkSession(sparkContext=sc)
    n = dataframe.count()
    bc_n = sc.broadcast(n)
    dict_missing_labels = _compute_fraction(
        sc=sc, dataframe=dataframe, label_col=label_col, fraction=fraction, **kwargs)

    # create the new schema
    schema = dataframe.schema.add('id', data_type=T.IntegerType(), nullable=False)

    # add id to dataframe
    rdd = dataframe.rdd.zipWithIndex().map(lambda x: (*x[0], x[1]))
    new_data_frame = spark.createDataFrame(rdd, schema=schema)
    labeled_id = new_data_frame.sampleBy(label_col, fractions=dict_missing_labels).select('id').collect()

    bc_labels = sc.broadcast(list(map(lambda x: x['id'], labeled_id)))
    return new_data_frame.withColumn(
        colName='missing_'+label_col, col=F.when(
            condition=F.col('id').isin(bc_labels.value),
            value=F.col(label_col)).otherwise(float('NAN'))
    )


def _compute_fraction(sc, dataframe, label_col, fraction=None, **kwargs):
    if fraction:  # set all labels to same fraction
        bcast = sc.broadcast(fraction)
        dict_missing_labels = (dataframe
            .groupBy(label_col).count().rdd
            .map(lambda x: (x['label'], bcast.value)).collectAsMap())
    else:
        dict_missing_labels = kwargs
    return dict_missing_labels


def distort_dataset(sc, dataframe, feature_cols):
    return dataframe



