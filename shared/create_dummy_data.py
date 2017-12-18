"""
Created on June, 2017

@author: sidselsrensen
"""
from pyspark import sql
import functools
import itertools
import pandas as pd
import numpy as np
import math

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

def create_spark_data(sc, func ,**kwargs):
    spark = sql.SparkSession(sparkContext=sc)
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    return spark.createDataFrame(func(**kwargs))

def export_csv(data_frame : sql.DataFrame,
               path='/home/svanhmic/workspace/data/DABAI/sparkdata/csv/double_helix.csv'):
    return data_frame.write.csv(path=path, mode='overwrite',header=data_frame.columns)

def create_double_helix(points_pr_helix, alpha=1.0, beta=1.0, missing = 0.01 ):
    if isinstance(missing, float) or missing < 1.0:
        missing = int(math.ceil(points_pr_helix*missing))

    x = np.random.uniform(0., 2*3.1415, points_pr_helix)
    double_helix = []
    for i, a in zip(range(2), [alpha, -alpha]):
        double_helix.append(list(map(lambda v: (a*np.sin(v), a*np.cos(v), beta*v, i), x)))

    pdf = pd.DataFrame(np.vstack(double_helix), columns='x y z label'.split(' '))
    unknown_label = lambda x: np.random.permutation([x]*missing+[np.nan]*(points_pr_helix-missing))

    pdf['unknown_label'] = np.hstack((unknown_label(0.0), unknown_label(1.0)))
    pdf['id'] = pdf.index
    return pdf

def load_mnist(n_samples = None, **kwargs):
    """
    Creates a dataframe with mnist data
    :param n_samples: extra parameter that enables extra digits
    :return:
    """
    path = kwargs.get('path','/home/svanhmic/workspace/data/DABAI/mnist')
    train_pdf = pd.read_csv(path+'/train.csv', header=0)
    test_pdf = pd.read_csv(path+'/test.csv',header=0)
    return train_pdf, test_pdf

