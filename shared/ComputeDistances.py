'''
Created on May 15, 2017

@author: svanhmic
'''
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import math


def compute_distance(point, center):
    '''
    Computes the euclidean  distance from a data point to the cluster center.
    
    :param point: coordinates for given point
    :param center: cluster center
    :return: distance between point and center
    '''
    assert isinstance(point, np.ndarray), str(point)+" is not an a numpy array"
    assert isinstance(center, np.ndarray), str(center)+" is not an a numpy array"

    squared_dist = np.dot(subtract_vectors(center-point), subtract_vectors(center-point))

    return math.sqrt(squared_dist)


def make_histogram(dist):
    ddist = [i["distance"] for i in dist.collect()]
    sns.distplot(ddist, rug=True)
    plt.show()


def subtract_vectors(vector_a: np.ndarray, vector_b: np.ndarray):
    '''
    Subtracts two numpy vectors from each other
    :param vector_a: numpy ndarray
    :param vector_b: numpy ndarray
    :return: numpy ndarray with subtracted vectors
    '''
    return vector_a - vector_b


