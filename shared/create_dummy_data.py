"""
Created on June, 2017

@author: sidselsrensen
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Row, SQLContext
from pyspark import SparkContext
from random import random
from string import digits, ascii_uppercase
from random import choice
import sys
import itertools

sqlCont = SQLContext.getOrCreate(sc=SparkContext.getOrCreate())


def create_dummy_data(number_of_samples, feature_names, label_names, **kwargs):
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
