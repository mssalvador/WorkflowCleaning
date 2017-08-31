from matplotlib.patches import Ellipse
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_gaussians(data_frame, featuresCol=None, predictionCol='prediction', clusterCol='centers', covarianceCol='cov',
                   **kwargs):
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
    feat_1 = featuresCol[0]
    feat_2 = featuresCol[1]
    pd_centers = data_frame.select(predictionCol, clusterCol, covarianceCol).distinct().toPandas()
    pd_centers[[feat_1, feat_2]] = pd.DataFrame([x for x in pd_centers[clusterCol]])
    pd_points = data_frame.select(feat_1, feat_2, predictionCol).toPandas()

    fig = plt.figure(figsize=(10, 6))
    pallet = sb.hls_palette(len(pd_centers), l=.4, s=.7)
    sb.set_palette(pallet)

    sb.lmplot(x=feat_1, y=feat_2, data=pd_points, fit_reg=False, hue=predictionCol
              , scatter_kws={'alpha': 0.4, 's': 60})

    for index, pd_cov_mean in pd_centers.iterrows():
        plot_cov_ellipse(cov=pd_cov_mean[covarianceCol].toArray(),
                         pos=pd_cov_mean[clusterCol],
                         nstd=gaussian_std,
                         alpha=0.5,
                         color=pallet[pd_cov_mean[predictionCol]])

    sb.regplot(x=feat_1,
               y=feat_2,
               data=pd_centers,
               fit_reg=False,
               marker='x',
               color='k',
               scatter_kws={"s": 100})

    plt.show()