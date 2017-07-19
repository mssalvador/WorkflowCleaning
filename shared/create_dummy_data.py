'''
Created on June, 2017

@author: sidselsrensen
'''

from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkContext
from random import random
from pyspark.sql.types import FloatType, StructType, StructField, StringType
from string import digits, ascii_uppercase
from random import choice
import sys

sqlCont = SQLContext.getOrCreate(sc=SparkContext.getOrCreate())


def create_dummy_data(number_of_samples, **kwargs):
    r'''
    :param number_of_samples
        number of rows in the Pyspark.dataframe
    :type 'int'
    :param \**kwargs
        see below
        
    :keyword 
     * *labels* ('dict') --
      label columns
    * *features* ('dict') -- 
      feature columns
    * *outlier_number* ('float') or ('int') --
      number of outliers in the in the Pyspark.dataframe; 
       either in procentage of total Pyspark.dataframe as float,
       or integer for number of rows in total 
    * *outlier_factor* ('int') --
      large number being multiplied on random() 
      to make outliers in the data
    
    :return: Pyspark.dataframe of size 'number_of_samples', with 'number_of_outliers' outliers of a 'outlier_factor'  
    '''

    labels = kwargs.get("labels", [])
    features = kwargs.get("features", [])
    outlier_number = kwargs.get("outlier_number", 0)
    outlier_factor = kwargs.get("outlier_factor", 0)

    # if neither labels nor features are found, the program will stop and show an error message
    if labels == [] and features == []:
        sys.exit("You must provide at least labels or features as a dictionary")

    if number_of_samples < outlier_number:
        sys.exit("Your total size needs to be bigger then your outlier size")

    if isinstance(outlier_number, float):
        outlier_number = int(outlier_number*number_of_samples)

    dummy_row = Row(*(labels+features))
    list_of_struct = []

    def structing(header, L=None):
        los = []
        if isinstance(header, list):
            for x in header:
                try:
                    los += [StructField(x, StringType()) if L else StructField(x, FloatType())]
                except:
                    print("{} must be string".format(header))
        return los

    def data_row(number_of_labels, number_of_features, factor=1):
        dr = []
        if number_of_labels != 0:
            dr.append(''.join([choice(ascii_uppercase+digits) for _ in range(8)]))
            for x in range(1, number_of_labels):
                dr.append('NR.'+''.join([choice(digits) for _ in range(4)]))
        if number_of_features != 0:
            for _ in range(number_of_features):
                dr.append(random()*factor)
        return dr

    def make_row_outlier(outlier_size, label_size, feature_size, factor=10):
        outliers = [data_row(label_size, feature_size, factor) for _ in range(outlier_size)]
        return outliers

    list_of_struct += structing(labels, L=1) + structing(features)
    schema = StructType(list_of_struct)

    dummy_data = [data_row(len(labels), len(features)) for _ in range(number_of_samples - outlier_number)]
    dummy_df = sqlCont.createDataFrame(list(map(lambda x: dummy_row(*x), dummy_data)), schema)

    if outlier_number > 0:
        outlier_data = make_row_outlier(outlier_number, len(labels), len(features), outlier_factor)
        outlier_df = sqlCont.createDataFrame(list(map(lambda x: dummy_row(*x), outlier_data)), schema)

        dummy_df_with_outliers = dummy_df.union(outlier_df)
        return dummy_df_with_outliers
    else:
        return dummy_df


def make_outliers(df, number_of_outliers, outlier_factor, **kwargs):
    '''
    :param df: Pyspark.dataframe that needs outliers
    :param number_of_outliers: integer or float
    :param outlier_factor: factor by which the columns are multiplied to create outliers
    :param kwargs: if only specific columns need changed. 
        dict with "features"
    :return: A modified Pyspark.dataframe with outlier data
    '''

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
    df = outlier_df.union(df_split[1])

    return df


