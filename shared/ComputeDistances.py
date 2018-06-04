"""
Created on May 15, 2017
@author: svanhmic
"""


def compute_distance(point, center):
    """
    Computes the euclidean  distance from a data point to the cluster center.
    :param point: coordinates for given point
    :param center: cluster center
    :return: distance between point and center
    """
    import numpy as np
    from pyspark.ml.linalg import SparseVector
    if isinstance(point, SparseVector) | isinstance(center, SparseVector):
        p_d = point.toArray()
        c_d = center.toArray()
        return float(np.linalg.norm(p_d-c_d, ord=2))
    else:
        return float(np.linalg.norm(point - center, ord=2))


def make_histogram(dist: list):  # , dim):
    """
    Makes the histogram of
    :param dist: Spark data frame  TODO: make this a list input instead
    :param dim: number of _dimensions that needs to be plotted
    :return:
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # isolate the distances from the data frame
    set_of_distances = set(dist)
    fig = plt.figure()
    if len(set_of_distances) > 1:
        sns.distplot(
            dist,
            rug=True,
            kde=True,
            norm_hist=False,
            # ax=ax)
        );
        fig.canvas.draw()
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




