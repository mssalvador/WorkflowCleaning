from matplotlib.patches import Ellipse
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(
        xy=pos,
        width=width,
        height=height,
        angle=theta,
        **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_gaussians(data_frame, featuresCol=None, predictionCol='prediction',
                   clusterCol='centers', covarianceCol='cov', **kwargs):
    """
    Creates a full plot with Gaussians mixtures, centers marked, elipsis' and data points
    :param data_frame: pyspark dataframe
    :param featuresCol: feature column
    :param predictionCol: prediction column
    :param clusterCol: cluster center column
    :param covarianceCol: covariance column
    :param kwargs: additional arguments
    :return: None, a plot...
    """
    assert len(featuresCol) == 2, 'This is a 2-D plot, number of features must be two, not ' + str(len(featuresCol))

    gaussian_std = kwargs.get('gaussian_std', 2)
    fig_size = kwargs.get('fig_size',7)
    feat_1 = featuresCol[0]
    feat_2 = featuresCol[1]

    if isinstance(data_frame, pd.DataFrame):
        try:
            pd_centers = kwargs.pop('pandasMeanCov')
        except NameError as ne:
            print(ne.with_traceback())
            return

        pd_points = data_frame[[feat_1, feat_2, predictionCol]]

    else:
        pd_centers = (data_frame
                      .select(predictionCol, clusterCol, covarianceCol)
                      .distinct()
                      .toPandas())

        pd_points = data_frame.select(feat_1, feat_2, predictionCol).toPandas()

    pd_centers[[feat_1, feat_2]] = pd.DataFrame([x for x in pd_centers[clusterCol]])

    #fig = plt.figure(figsize=(10, 6))
    pallet = sb.hls_palette(len(pd_centers), l=.4, s=.7)
    sb.set_palette(pallet)

    sb.lmplot(
        x=feat_1,
        y=feat_2,
        data=pd_points,
        fit_reg=False,
        size=fig_size,
        hue=predictionCol,
        scatter_kws={'alpha': 0.4, 's': 60})

    for index, pd_cov_mean in pd_centers.iterrows():
        plot_cov_ellipse(
            cov=pd_cov_mean[covarianceCol].toArray(),
            pos=pd_cov_mean[clusterCol],
            nstd=gaussian_std,
            alpha=0.5,
            color=pallet[pd_cov_mean[predictionCol]])

    sb.regplot(
        x=feat_1,
        y=feat_2,
        data=pd_centers,
        fit_reg=False,
        marker='x',
        color='k',
        scatter_kws={"s": 100})

    plt.show()

def plot_known_and_unknown_data(pdf, x='x', y='y', labelCol='used_label', **kwargs):
    """
    Used to plot semi supervised learning data, either a full dataset where all
    lables are set or an incomplete dataset where a portion of the data is set.

    @input: pdf: pandas dataframe with all data. Must contain a label column and feature
    @input: x: first feature
    @input: y: second feature
    @input: labelCol: the name of the label column must be known!
    @input: **kwargs: arguments like plot titel.
    """
    fig, ax = plt.subplots(1, figsize=(15, 15))
    clusters = list(filter(lambda x: x != np.nan, pdf[labelCol].dropna().unique()))
    pallet = sb.hls_palette(len(clusters) + 1, l=.4, s=.7)
    sb.set_palette(pallet)
    label = list(map(lambda x: 'Cluster {}'.format(int(x)), clusters))

    unknown_pdf = pdf[pdf[labelCol].isnull() == True]
    symbols = ['o', '<']

    for idx, k in enumerate(clusters):
        ax.plot(unknown_pdf[unknown_pdf['real_label'] == k][x],
                unknown_pdf[unknown_pdf['real_label'] == k][y],
                symbols[idx],
                color=pallet[-1],
                alpha=0.60,
                label='Unlabled_data')

        ax.plot(pdf[pdf[labelCol] == k][x],
                pdf[pdf[labelCol] == k][y],
                symbols[idx],
                color=pallet[int(k)],
                label=label[int(k)]
                )
    ax.set_title(kwargs.get('title', 'Plot of dataset with and without lables'), fontsize=30)
    ax.legend(loc=0)
    plt.show()

def plot3D(data, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    label = kwargs.get('label','label')

    if not isinstance(data, pd.DataFrame):
        pandas_transition = data.toPandas()
    else:
        pandas_transition = data

    pandas_transition= pandas_transition.fillna(-1)
    pandas_columns = lambda x, pdf: [
        pdf[pdf[label] == x][dim] for dim in 'x y z'.split(' ')]

    for i in pandas_transition[~pandas_transition[label].isnull()][label].unique():
        ax.scatter(*pandas_columns(i, pandas_transition))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Our test dataset a double helix')
    plt.show()