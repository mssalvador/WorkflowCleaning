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

    squared_dist = np.dot((center-point), (center-point))

    return math.sqrt(squared_dist)


def make_histogram(dist):
    ddist = [i["distance"] for i in dist.collect()]
    sns.distplot(ddist, rug=True)
    plt.show()


