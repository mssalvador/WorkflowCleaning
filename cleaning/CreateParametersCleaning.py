"""
@author: svanhmic
@date: 21 august 2017
@description: Create a widget page in a class, for selecting parameters for classification
"""

from ipywidgets import widgets
from traitlets import dlink
from pyspark.ml import clustering
import logging
import sys
from pyspark import SparkContext
import random
from shared import OwnFloatSingleSlider, OwnCheckBox, OwnIntSingleSlider, OwnSelect, OwnText, OwnDropdown

# setup logging first, better than print!
logger_parameter_select = logging.getLogger(__name__)
logger_parameter_select.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_cleaning.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_parameter_select.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)

sc = SparkContext.getOrCreate()


class ParamsCleaning(object):
    """
    This class is to create a parameter map for the cleaning algorithms.

    """
    algorithm_clustering = [str(i) for i in clustering.__all__
                            if ("Model" not in i) if ("Summary" not in i) if ("BisectingKMeans" not in i)]

    def __init__(self):
        logger_parameter_select.info(" Create_Cleaning_Parameters created")

        self._selected_parameters = {"algorithm": self.algorithm_clustering[0]}
        self._algorithms_and_paramters = ParamsCleaning.create_parameters()

    def __repr__(self):
        return "ParamsClustering()"

    def __str__(self):
        return '{}'.format(self._selected_parameters.get("algorithm", self.__repr__()))

    @staticmethod
    def output_parameters(params):
        return dict([(x.name, x.value) for l in params.children for x in l.children])

    @classmethod
    def create_parameters(cls):
        '''
        Initial method for creating all _parameters for all algorithms along with default vals
        :return:
        '''

        algo_and_params = {}
        for i in cls.algorithm_clustering:
            model = getattr(clustering, i)()
            maps = model.extractParamMap()

            algo_and_params[i] = dict(zip(map(lambda x: x.name, maps.keys()), maps.values()))
            logger_parameter_select.debug(
                " Parameters selected for algorithm {} with _parameters {}".format(i, algo_and_params[i]))
        return algo_and_params

    def select_parameters(self):

        '''
        The main method for selecting _parameters to each algorithm. Each algorithm has its own set of _parameters.
        :return: None
        '''
        widget_dropdown_algorithms = OwnDropdown.OwnDropdown(
            options=ParamsCleaning.algorithm_clustering,
            value=ParamsCleaning.algorithm_clustering[0],
            description="Algorithms",
            disabled=False,
            name="algorithm")

        all_widgets = widgets.VBox([widget_dropdown_algorithms])

        list_of_methods = [self.create_kmeans_clustering_widgets,
                           self.create_gaussian_mixture_widgets,
                           self.create_lda_clustering_widgets]

        widget_and_algorithms = dict(zip(ParamsCleaning.algorithm_clustering, list_of_methods))

        def update_algorithm_parameters(change):

            if change in widget_and_algorithms.keys():
                logger_parameter_select.debug(" Algorithm changed to: {}".format(change))
                return widget_and_algorithms[change](self._algorithms_and_paramters[change])
            else:
                raise NotImplementedError

        dlink((widget_dropdown_algorithms, 'value'),
              (all_widgets, "children"),
              lambda val: [widgets.HBox([widget_dropdown_algorithms])] + update_algorithm_parameters(val))

        return all_widgets

    @staticmethod
    def create_kmeans_clustering_widgets(dict):
        """
        instiantate the widgets for kmeans clustering algortihm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """
        widget_k = OwnIntSingleSlider.OwnIntSingleSlider(
            value=dict.get("n_clusters", 10),
            min=2,
            max=200,
            step=1,
            description="Number of Clusters",
            name="n_clusters")

        widget_initMode = OwnDropdown.OwnDropdown(
            value=dict.get("initMode", "n_clusters-means||"),
            options=["n_clusters-means||",  "random"],
            description="Initial mode",
            name="initMode")

        widget_initSteps = OwnIntSingleSlider.OwnIntSingleSlider(
            value=dict.get("initSteps", 10),
            min=1,
            max=50,
            step=1,
            description="Number of Initial steps",
            name="initSteps")

        widget_tol = OwnFloatSingleSlider.OwnFloatSingleSlider(
            value=dict.get("tol", 1e-4),
            min=1e-4,
            max=1e-3,
            step=1e-4,
            description="Tolerance",
            name="tol")

        widget_maxIter = OwnIntSingleSlider.OwnIntSingleSlider(
            value=dict.get("maxIter", 100),
            min=10,
            max=200,
            step=1,
            description="Max iterations",
            name="maxIter")

        widget_seed = OwnIntSingleSlider.OwnIntSingleSlider(
            value=dict.get("seed", random.randint(0, sys.maxsize)),
            min=0,
            max=sys.maxsize,
            step=1000,
            description="Seed",
            name="seed")

        all_lists = [[widget_k, widget_initSteps, widget_tol],
                     [widget_maxIter, widget_seed, widget_initMode]]
        return list(map(lambda x: widgets.HBox(x), all_lists))

    @staticmethod
    def create_gaussian_mixture_widgets(dict):
        """
        instiantate the widgets for Gausian mixture models algortihm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """

        widget_k = OwnIntSingleSlider.OwnIntSingleSlider(value=dict.get("n_clusters", 10),
                                                         min=2,
                                                         max=200,
                                                         step=1,
                                                         description="Number of Clusters",
                                                         name="n_clusters")

        widget_tol = OwnFloatSingleSlider.OwnFloatSingleSlider(value=dict.get("tol", 1e-4),
                                                               min=1e-4,
                                                               max=1e-3,
                                                               step=1e-4,
                                                               description="Tolerance",
                                                               name="tol")

        widget_maxIter = OwnIntSingleSlider.OwnIntSingleSlider(value=dict.get("maxIter", 100),
                                                               min=10,
                                                               max=200,
                                                               step=1,
                                                               description="Max iterations",
                                                               name="maxIter")

        widget_seed = OwnIntSingleSlider.OwnIntSingleSlider(value=dict.get("seed", random.randint(0, sys.maxsize)),
                                                            min=0,
                                                            max=sys.maxsize,
                                                            step=1000,
                                                            description="Seed",
                                                            name="seed")

        all_lists = [[widget_k, widget_maxIter],
                     [widget_tol, widget_seed]]
        return list(map(lambda x: widgets.HBox(x), all_lists))

    @staticmethod
    def create_lda_clustering_widgets(dict):
        """
        instiantate the widgets for LDA clustering algortihm

        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        """
        pass
