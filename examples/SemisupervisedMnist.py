from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from semisupervised.labelpropagation import label_propagation
import functools


def subset_dataset_by_label(sc, dataframe : DataFrame, label_col, *args):
    if args:
        return dataframe.filter(F.col(label_col).isin(list(args)))
    else:
        return dataframe


def enlarge_dataset(dataframe : DataFrame , size=None, feature_cols=None, label_col=None, **kwargs):
    # size must be larger than dataframe size
    n = dataframe.count()
    if n <= size:
        extras = (size-n)/n
        extra_df = dataframe.sample(withReplacement=True, fraction=extras)
        columns = [F.col(i) if i not in feature_cols else F.col(i)+F.rand() for i in extra_df.columns]
        return extra_df.select(columns).union(dataframe)
    else:
        frac = 1-(n-size)/n
        samples = dataframe.sample(withReplacement=False, fraction=frac)
        all_types = list(map(lambda x: x[label_col], samples.select(label_col).distinct().collect()))
        while not (len(list(all_types)) == kwargs.get('k', 10)):
            print('iteration')
            samples = dataframe.sample(withReplacement=False, fraction=frac)
            all_types = list(map(lambda x: x[label_col], samples.select(label_col).distinct().collect()))
        return samples


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
    # print(dict_missing_labels)
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


def run_experiment(sc, dataframe, label_col, feature_cols=None, data_size=None, fractions=None, **kwargs):

    # Increase or decrese size
    if data_size:
        sized_df = enlarge_dataset(
            dataframe=dataframe, size=data_size,
            feature_cols=feature_cols, label_col=label_col, **kwargs)
    else:
        sized_df = dataframe

    # Make dataset with nan values
    added_nan_df = create_nan_labels(
        sc=sc, dataframe=sized_df, label_col=label_col, fraction=fractions, **kwargs)

    partial_lp = functools.partial(
        label_propagation, sc=sc, data_frame=added_nan_df,
        label_col='missing_'+label_col, id_col='id',
        feature_cols=feature_cols)
    output_data_frame = partial_lp(**kwargs)
    return output_data_frame




