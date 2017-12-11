"""
Created on June, 2017

@author: sidselsrensen
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
import functools
import itertools
import pandas as pd
import numpy as np

def create_norm_cluster_data_pandas(n_amounts, means, std=None, features=None):
    """
    Creates an n*m dimensional dataframe with normal distributed data
    @input:
    @input: means: a n_clusters-long list containing n_dimension-dimensional points acting as means
    @input: std: a n_clusters-long list containgin n_dimension-dimensional standard deviation for the normal distribution
    @input: feature_names: list containing names for each column
    @return: clusters: pandas dataframe with n_clusters clusters and amounts_k number of data points pr cluster
    """
    assert len(means) == len(n_amounts), "number of means is different from number of clusters"
    if isinstance(features, int):
        features = list(map(chr, range(ord('a'), ord('a') + features, 1)))  # generate a number of features
    elif isinstance(features, list):
        features = features
    else:
        features = list(map(chr, range(ord('a'), ord('a')+10, 1))) # generate a number of features

    if not std:
        k = len(n_amounts)
        std = np.ones((k, len(features)))

    X = [np.random.normal(
        means[elements], std[elements], [n_amounts[elements], len(features)]) for elements in range(0, len(n_amounts))]

    data_frame = pd.DataFrame(np.vstack(X), columns=features)
    data_frame['id'] = np.array(range(0, functools.reduce(lambda a, b:a+b, n_amounts), 1))
    data_frame['k'] = np.array(list(itertools.chain(
        *[ns*[ks] for ns, ks in zip(n_amounts, range(len(n_amounts)))])))
    return data_frame

def create_norm_cluster_data_spark(sc, n_amounts, means, std=None, features=None):
    spark = SparkSession(sparkContext=sc)
    return spark.createDataFrame(create_norm_cluster_data_pandas(
        n_amounts=n_amounts, means=means, std=std, features=features))
