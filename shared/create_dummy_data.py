"""
Created on June, 2017

@author: sidselsrensen
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Row, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark import SparkContext
from random import random
from string import digits, ascii_uppercase
from random import choice
import sys
import itertools
import pandas as pd
import numpy as np

def create_dummy_data(sc, number_of_samples, feature_names, label_names, **kwargs):
    r"""
    Create a dataframe that contains outliers

    :param number_of_samples: Number of rows in the Pyspark.dataframe :type 'int'
    :param feature_names: A list of feature_names names
    :param label_names: A list of label names
    :param kwargs: see below
    :keyword
     * *labels* ('dict') --
      label columns
    * *feature_names* ('dict') --
      feature columns
    * *outlier_number* ('float') or ('int') --
      number of outliers in the in the Pyspark.dataframe;
       either in procentage of total Pyspark.dataframe as float,
       or integer for number of rows in total
    * *outlier_factor* ('int') --
      large number being multiplied on random()
      to make outliers in the data

    :return: Pyspark.dataframe of size 'number_of_samples', with 'number_of_outliers' outliers of a 'outlier_factor'
    """

    #label_names = kwargs.get("label_names", [])
    #feature_names = kwargs.get("feature_names", [])
    sqlCont = SQLContext.getOrCreate(sc=sc)
    outlier_number = kwargs.get("outlier_number", 0)
    outlier_factor = kwargs.get("outlier_factor", 0)

    # If neither label_names nor feature_names are found, the program will raise an ValueError message
    label_names, feature_names = check_input_feature_label(label_names, feature_names)

    if number_of_samples < outlier_number:
        sys.exit("Your total size needs to be bigger then your outlier size")

    if isinstance(outlier_number, float):
        outlier_number = int(outlier_number*number_of_samples)

    # Create the data frame schema
    struct_label = map(lambda x: structing(x, True), label_names)    # Create the structFields for Labels
    struct_feat = map(lambda x: structing(x), feature_names)         # Create the structFields for Labels
    schema = T.StructType(list(itertools.chain(struct_label, struct_feat)))

    # Create custom row-class, aka. give the df a header
    dummy_row = Row(*(label_names + feature_names))

    # Create the data frame without outliers
    dummy_data = [data_row(len(label_names), len(feature_names)) for _ in range(number_of_samples - outlier_number)]
    dummy_df = sqlCont.createDataFrame(list(map(lambda x: dummy_row(*x), dummy_data)), schema)

    # Create a data frame with outliers and union and return
    if outlier_number > 0:
        outlier_data = make_row_outlier(outlier_number, len(label_names), len(feature_names), outlier_factor)
        outlier_df = sqlCont.createDataFrame(list(map(lambda x: dummy_row(*x), outlier_data)), schema)
        return dummy_df.union(outlier_df)
    else:
        return dummy_df


def make_outliers(df, number_of_outliers, outlier_factor, **kwargs):
    """
    Create outliers in data frame.
    :param df: Pyspark.dataframe that needs outliers
    :param number_of_outliers: integer or float
    :param outlier_factor: factor by which the columns are multiplied to create outliers
    :param kwargs: if only specific columns need changed.
        dict with "features"
    :return: A modified Pyspark.dataframe with outlier data
    """

    features = kwargs.get("features", [f[0] for f in df.dtypes if f[1] == 'float'])
    if isinstance(number_of_outliers, float):
        number_of_outliers = number_of_outliers*df.count()

    df_split = df.randomSplit([float(number_of_outliers), float(df.count() - number_of_outliers)])
    if df_split[0]:
        while not number_of_outliers*0.9 <= df_split[0].count() <= number_of_outliers*1.1:
            df_split = df.randomSplit([float(number_of_outliers), float(df.count() - number_of_outliers)])

    outlier_feat = [(F.col(feat) * outlier_factor).alias(feat) for feat in features]
    other_feat = [i for i in df.schema.names if i not in features]

    outlier_df = df_split[0].select(other_feat + outlier_feat)
    return outlier_df.union(df_split[1])


def check_input_feature_label(label, feature):
    """
    Simple input checker for features and labels
    :param label: A list or string of label names
    :param feature: A list or string of features names
    :rtype list:
    """
    if not label:
        raise ValueError("Label must contain at least 1 value.")
    elif not feature:
        raise ValueError("You must provide at least 1 feature.")
    else:
        return make_list(label), make_list(feature)


def make_list(input_list):
    """
    Checks if the list is either a list or string.
    :param input_list:
    :rtype list
    """

    if isinstance(input_list, str):
        return input_list.split()
    else:
        return input_list


def structing(header, L=False):
    """
    Creates a StructField with either a StringType or a FloatType
    :param header: variable name
    :param L: bool if True/1 it's a StringType else Floattype
    :return: StructField()
    """

    try:
        return T.StructField(header, T.StringType()) if L else T.StructField(header, T.DoubleType())
    except:
        print("{} must be string".format(header))
        return T.StructField(str(header), T.StringType()) if L else T.StructField(str(header), T.DoubleType())


def data_row(number_of_labels, number_of_features, factor=1):
    """
    Create random data rows that aren't outliers
    :param number_of_labels:
    :param number_of_features:
    :param factor:
    :return:list of list with random data in both label and features
    """
    list_data_row = []
    if number_of_labels != 0:
        list_data_row.append(''.join([choice(ascii_uppercase+digits) for _ in range(8)]))
        for x in range(1, number_of_labels):
            list_data_row.append('NR.'+''.join([choice(digits) for _ in range(4)]))
    if number_of_features != 0:
        for _ in range(number_of_features):
            list_data_row.append(random()*factor)
    return list_data_row


def make_row_outlier(outlier_size, label_size, feature_size, factor=10):
    return [data_row(label_size, feature_size, factor) for _ in range(outlier_size)]


def create_normal_cluster_data_pandas(amounts, means, std=None, labels=None):
    """
    Creates a dataframe with normal data
    @input: means: a n_clusters-long list containing n_dimension-dimensional points acting as means
    @input: std: a n_clusters-long list containgin n_dimension-dimensional standard deviation for the normal distribution
    @input: labels: list containing names for each column
    @return: clusters: pandas dataframe with n_clusters clusters and amounts_k number of data points pr cluster
    """

    import pandas as pd
    import numpy as np

    result = pd.DataFrame(columns=labels)

    assert len(means) == len(amounts), "number of means is different from number of clusters"

    dim = len(amounts[0])
    if std is None:
        k = len(amounts)
        std = np.ones((k, dim))

    if labels is None:
        labels = list(map(chr, range(ord('a'), ord('a') + dim, 1)))

    for k, n in enumerate(amounts):
        x = np.random.normal(means[k], std[k], size=n)
        result = result.append(pd.DataFrame(x, columns=labels))
    return result


def create_normal_cluster_data_spark(sc, dim, n_samples, means, std):
    """
    Create a Spark dataframe that with clusters
    :param dim: dimension in data
    :param n_samples: list containing number of samples pr cluster
    :param means: list with cluster mean, dimension must fit
    :param std: list with cluster std, dimension must fit
    :return: spark dataframe with clusters
    """

    import numpy as np
    sqlCont = SQLContext.getOrCreate(sc=sc)
    # create the fixed schema labels
    schema_fixed = [T.StructField('id', T.IntegerType()),
                    T.StructField('n_clusters', T.IntegerType()),
                    T.StructField('dimension', T.IntegerType())]

    # create the moving schema labels
    label_names = list(map(chr, range(ord('a'), ord('a') + dim, 1)))
    schema_dimension = [T.StructField(i, T.DoubleType()) for i in label_names]
    schema = T.StructType(schema_fixed + schema_dimension)

    # broadcasts
    broadcast_mean = sc.broadcast(dict(enumerate(means)))
    broadcast_std = sc.broadcast(dict(enumerate(std)))

    # create the normal distributed data point.
    def create_arr(k, dim):
        return [float(i) for i in np.random.normal(
            broadcast_mean.value[k],
            broadcast_std.value[k],
            (1, dim))[0]]

    # make it into an udf
    udf_rand = F.udf(lambda k, d: create_arr(k, d), T.ArrayType(T.DoubleType()))

    # create the dataframe in steps
    result_df = sqlCont.createDataFrame(sc.emptyRDD(), schema)

    for k, n in enumerate(n_samples):
        # print(n_clusters)
        # print(n)
        cols = ['id', 'n_clusters', 'dimension'] +\
               [F.col('vec')[i].alias(str(i)) for i in range(dim)]

        df = (sqlCont
              .range(0, n, 1)
              .withColumn('dimension', F.lit(dim))
              .withColumn('n_clusters', F.lit(k))
              .withColumn('vec', udf_rand('n_clusters', 'dimension'))
              .select(cols)
              )
        result_df = result_df.union(df)
    return result_df


def create_partition_samples(n, k):
    import math
    import numpy as np
    arr = np.array(list(map(lambda x: math.floor(x*n), np.random.dirichlet(np.ones(k)))))
    if sum(arr) != n:
        arr[np.random.randint(0, k, 1)[0]] += n-sum(arr)
    return arr


def create_means(dim, k, factor):
    import numpy as np
    return [np.random.uniform(low=-factor, high=factor, size=dim) for _ in range(k)]


def create_stds(dim, k, factor=1):
    import numpy as np
    return [factor*np.ones(dim) for _ in range(k)]


def create_labeled_data_with_clusters(n, mean, std, frac_label=0.1):
    """
    Creates a dataset with k clusters surrounding k means in 2D
    @param: n: int or list with k values for each cluster
    @param: mean: int or list with k values of mean for each cluster
    @param: std: int or list with k values of std for each cluster
    @param: frac_label: fraction of labels  that are not deleted
    @return: pandas dataset with clusters
    """

    cols = ['x', 'y', 'real_label', 'used_label']

    if isinstance(n, int) and isinstance(mean[0], float):
        x = np.random.normal(loc=mean,
                             scale=std,
                             size=[n, 2])
        return pd.DataFrame({
            cols[0]: x[:, 0],
            cols[1]: x[:, 1],
            cols[2]: np.zeros(n),
            cols[3]: float(0) * np.ones(n_val)}
        )
    else:
        matrix = pd.DataFrame(columns=cols)
        for idx, n_val in enumerate(n):
            x = np.random.normal(loc=mean[idx],
                                 scale=std[idx],
                                 size=[n_val, 2])
            hidden_label_vec = float(idx) * np.ones((n_val, 1))
            x = np.concatenate((x, hidden_label_vec), axis=1)

            used_label_vec = create_vector_with_nan_vals(
                hidden_label_vec,
                fraction=frac_label)

            x = np.concatenate(
                (x, used_label_vec),
                axis=1)

            matrix = pd.concat(
                [matrix, pd.DataFrame(x, columns=cols)])
        return matrix


def create_vector_with_nan_vals(vector, fraction):
    nan_elements = int(len(vector) * (1.0 - fraction))
    if nan_elements == 0:
        nan_elements = 1
    indicies = np.random.choice(len(vector), nan_elements, replace=False)
    for i in indicies:
        vector[i] = np.NaN
    return vector


def create_contious_id_data(sc, input_path = None, output_path = None):

    spark_session = SparkSession(sc)
    win_spec = Window().orderBy('k')
    df_test_data = (spark_session
                    .read
                    .parquet(input_path)
                    .drop('dimension')
                    .withColumn(colName= 'label',
                                col= F.when(F.col('id') % 2 == 0, 1.0).otherwise(0.0))
                    .withColumn(colName= 'label',
                                col= F.when(F.col('id') < 2, F.col('label')).otherwise(None))
                    )

    cummalative_sum = (df_test_data
                       .groupBy('k')
                       .count()
                       .select(F.col('k'), F.sum('count').over(win_spec).alias('c_sum'))
                       .rdd
                       .map(lambda x: (x[0] + 1, x[1]))
                       .collectAsMap()
                       )

    cummalative_sum[0] = 0
    bcast_c_sum = sc.broadcast(cummalative_sum)
    c_sum_udf = F.udf(lambda k: bcast_c_sum.value[k], T.IntegerType())

    (df_test_data
     .withColumn('id', F.col('id') + c_sum_udf(F.col('k')))
     .orderBy('id')
     .drop('k')
     .write
     .parquet(path=output_path, mode= 'overwrite')
     )