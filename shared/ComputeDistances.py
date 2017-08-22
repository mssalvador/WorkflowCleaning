"""
Created on May 15, 2017

@author: svanhmic
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2


def compute_distance(point, center):
    """
    Computes the euclidean  distance from a data point to the cluster center.

    :param point: coordinates for given point
    :param center: cluster center
    :return: distance between point and center
    """
    assert isinstance(point, np.ndarray), str(point)+" is not an a numpy array"
    assert isinstance(center, np.ndarray), str(center)+" is not an a numpy array"

    squared_dist = np.dot((center-point), (center-point))
    return float(math.sqrt(squared_dist))


def make_histogram(dist: list, dim):

    """
    :param dist: Spark data frame  TODO: make this a list input instead
    :param dim: number of _dimensions that needs to be plotted
    :return:
    """

    # isolate the distances from the data frame
    set_of_distances = set(dist)

    if len(set_of_distances) > 1:
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(chi2.ppf(0.01, dim), chi2.ppf(0.99, dim), 100)
        ax.plot(x, chi2.pdf(x, dim), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
        sns.distplot(dist, rug=True, kde=True, norm_hist=False)
        plt.show()
    else:
        print('Too few datapoints to show')


def compute_percentage_dist(distance, max_distance):
    """

    :param max_distance:
    :param distance
    :return: distance between point and center
    """
    return float(max_distance-distance)/100


def make_components_histogram(agg_components, dimension):
    """
    Create a histogram of vector components for each cluster
    :param agg_components: Aggregated components perhaps even normalized
    :param dimension: labels containing name of each feature.
    :return: None. Displays a graph
    """



    pass


