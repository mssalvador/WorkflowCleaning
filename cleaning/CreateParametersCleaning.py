"""
@author: svanhmic
@date: 21 august 2017
@description: Create a widget page in a class, for selecting parameters for classification
"""

from ipywidgets import widgets
from traitlets import dlink

from pyspark.ml import clustering
import sys
from pyspark import SparkContext
import random
from shared.WorkflowLogger import logger_info_decorator

sc = SparkContext.getOrCreate()


class ParamsCleaning(object):
    """
    This class is to create a parameter map for the cleaning algorithms.

    """
    algorithm_clustering = [str(i) for i in clustering.__all__
                            if ("Model" not in i) if ("Summary" not in i) if ("BisectingKMeans" not in i)]

    @logger_info_decorator
    def __init__(self):
        self._selected_parameters = {"algorithm": self.algorithm_clustering[0]}
        self._algorithms_and_parameters = ParamsCleaning.create_parameters()

    def __repr__(self):
        return "ParamsClustering()"

    def __str__(self):
        return '{}'.format(self._selected_parameters.get("algorithm", self.__repr__()))

    @staticmethod
    def output_parameters(params):

        arr = [(x.name, x.value) for l in params.children for x in l.children]
        return dict(arr)

    @classmethod
    @logger_info_decorator
    def create_parameters(cls):
        """
        Initial method for creating all _parameters for all algorithms along with default vals
        :return:
        """

        algo_and_params = {}
        for i in cls.algorithm_clustering:
            model = getattr(clustering, i)()
            maps = model.extractParamMap()
            algo_and_params[i] = dict(zip(map(lambda x: x.name, maps.keys()), maps.values()))
        return algo_and_params

    def select_parameters(self):
        """
        The main method for selecting _parameters to each algorithm. Each algorithm has its own set of _parameters.
        :return: None
        """

        widget_dropdown_algorithms = widgets.Dropdown(
            options=ParamsCleaning.algorithm_clustering,
            value=ParamsCleaning.algorithm_clustering[0],
            description="Algorithms",
            disabled=False)
        setattr(widget_dropdown_algorithms,'name','algorithm')

        all_widgets = widgets.VBox([widget_dropdown_algorithms])

        list_of_methods = [self.create_kmeans_clustering_widgets,
                           self.create_gaussian_mixture_widgets,
                           self.create_lda_clustering_widgets]

        widget_and_algorithms = dict(zip(ParamsCleaning.algorithm_clustering, list_of_methods))

        @logger_info_decorator
        def update_algorithm_parameters(change):

            if change in widget_and_algorithms.keys():
                return widget_and_algorithms[change](self._algorithms_and_parameters[change])
            else:
                raise NotImplementedError

        (dlink((widget_dropdown_algorithms, 'value'),
               (all_widgets, "children"),
               lambda val: [widgets.HBox([widget_dropdown_algorithms])] + update_algorithm_parameters(val)))

        return all_widgets

    @staticmethod
    def create_kmeans_clustering_widgets(dict):
        """
        instiantate the widgets for k-means clustering algorithm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """

        widget_initMode = widgets.Dropdown(
            value=dict.get("initMode", "k-means||"),
            options=["k-means||",  "random"],
            description="Initial mode")
        setattr(widget_initMode, 'name', 'initMode')

        widget_initSteps = widgets.IntSlider(
            value=dict.get("initSteps", 10),
            min=1,
            max=100,
            step=1,
            description="Number of Initial steps")
        setattr(widget_initSteps, 'name', 'initSteps')

        widgets_gaussian_mix = ParamsCleaning.create_gaussian_mixture_widgets(dict)

        widgets_kmeans = widgets.HBox(
            [widget_initSteps, widget_initMode])

        return widgets_gaussian_mix + [widgets_kmeans]

    @staticmethod
    def create_gaussian_mixture_widgets(dict):
        """
        instiantate the widgets for Gaussian mixture models algorithm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """

        widget_k = widgets.IntSlider(
            value=dict.get("k", 10),
            min=2,
            max=200,
            step=1,
            description="Number of Clusters")
        setattr(widget_k, 'name', 'k')

        widget_tol = widgets.FloatSlider(
            value=dict.get("tol", 1e-4),
            min=1e-4,
            max=1e-3,
            step=1e-4,
            description="Tolerance")
        setattr(widget_tol, 'name', 'tol')

        widget_maxIter = widgets.IntSlider(
            value=dict.get("maxIter", 100),
            min=10,
            max=1000,
            step=1,
            description="Max iterations")
        setattr(widget_maxIter, 'name', 'maxIter')

        widget_seed = widgets.IntSlider(
            value=dict.get("seed", random.randint(0, sys.maxsize)),
            min=0,
            max=sys.maxsize,
            step=1000,
            description="Seed",
            )
        setattr(widget_seed, 'name', 'seed')

        all_lists = [[widget_k, widget_maxIter],
                     [widget_tol, widget_seed]]
        return list(map(lambda x: widgets.HBox(x), all_lists))

    @staticmethod
    def create_lda_clustering_widgets(dict):
        """
        instiantate the widgets for LDA clustering algorithm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """
        pass
