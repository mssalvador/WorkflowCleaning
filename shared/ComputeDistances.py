'''
Created on May 15, 2017

@author: svanhmic
'''
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2


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


def make_histogram(dist, dim):
    ddist = [i["distance"] for i in dist.collect()]

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(chi2.ppf(0.01, dim), chi2.ppf(0.99, dim), 100)
    ax.plot(x, chi2.pdf(x, dim), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
    sns.distplot(ddist, rug=True)

    plt.show()


