'''
Created on Jul 5, 2017

@author: svanhmic

@description: Create a widget page in a class, for selecting parameters for classification
'''


from ipywidgets import widgets
from functools import partial
from traitlets import dlink
from pyspark.ml import classification
import logging
from pyspark import SparkContext
from shared.WorkflowLogger import logger_info_decorator

sc = SparkContext.getOrCreate()


class ParamsClassification(object):

    algorithm_classification = [str(i) for i in classification.__all__
                                if ("Model" not in i) if ("Summary" not in i)]

    @logger_info_decorator
    def __init__(self):
        self._selected_parameters = {"algorithm": self.algorithm_classification[0]}
        self._algorithms_and_paramters = ParamsClassification.create_parameters()

    def __repr__(self):
        return "ParamsClassification()"

    def __str__(self):
        return '{}'.format(self._selected_parameters.get("algorithm", self.__repr__()))

    @staticmethod
    def output_parameters(params):
        return dict([(x.name, x.value) for l in params.children for x in l.children])

    @classmethod
    @logger_info_decorator
    def create_parameters(cls):
        '''
        Initial method for creating all _parameters for all algorithms along with default vals
        :return:
        '''

        algo_and_params = dict()

        for i in cls.algorithm_classification:
            model = getattr(classification, i)()
            maps = model.extractParamMap()

            algo_and_params[i] = dict(zip(map(lambda x: x.name, maps.keys()), maps.values()))
        return algo_and_params

    def select_parameters(self):

        '''
        The main method for selecting _parameters to each algorithm. Each algorithm has its own set of _parameters.
        :return: None
        '''
        widget_dropdown_algorithms = widgets.Dropdown(
            options=ParamsClassification.algorithm_classification,
            value=ParamsClassification.algorithm_classification[0],
            description="Algorithms",
            disabled=False)
        setattr(widget_dropdown_algorithms, 'name', 'algorithm')

        all_widgets = widgets.VBox([widget_dropdown_algorithms])

        list_of_methods = [self.create_logistic_regression_widgets,
                           self.create_decision_tree_classifier_widgets,
                           self.create_GB_tree_classifier_widgets,
                           self.create_random_forrest_classifier_widgets,
                           self.create_naive_bayes_widgets,
                           self.create_multi_layer_perception_widget,
                           self.create_one_vs_rest]

        widget_and_algorithms = dict(zip(ParamsClassification.algorithm_classification, list_of_methods))

        @logger_info_decorator
        def update_algorithm_parameters(change):

            if change in widget_and_algorithms.keys():
                return widget_and_algorithms[change](self._algorithms_and_paramters[change])
            else:
                raise NotImplementedError

        dlink((widget_dropdown_algorithms, 'value'), (all_widgets, "children"),
              lambda val: [widgets.HBox([widget_dropdown_algorithms])]+update_algorithm_parameters(val))

        return all_widgets

    @staticmethod
    def create_logistic_regression_widgets(dict):
        '''
        initiate the logistic regression widgets
        :param dict: dictionary with _parameters for logistic regression along with default values
        :return: list with HBox's of widgets
        '''
        widget_temp_float_range_slider = partial(
            widgets.FloatRangeSlider,
            min=0.0,
            max=1.0,
            step=0.01)

        widget_elasticNetParam = widget_temp_float_range_slider(
            value= (dict.get("elasticNetParam", 0.0), 0.5),
            description="Elastic Net")
        setattr(widget_elasticNetParam, 'name', 'elasticNetParam')

        widget_regParam = widget_temp_float_range_slider(
            value=(dict.get("regParam", 0.0), 0.5),
            description="Regularization")
        setattr(widget_regParam, 'name', 'regParam')

        widget_threshold = widget_temp_float_range_slider(
            value=(0.0, dict.get("threshold", 0.5)),
            description="Threshold")
        setattr(widget_threshold, 'name', 'threshold')

        widget_tol = widget_temp_float_range_slider(
            value=(dict.get("tol", 0.001), 0.01),
            description="Tolerance")
        setattr(widget_tol, 'name', 'tol')


        widget_fitIntercept = widgets.Checkbox(
            value=dict.get("fitIntercept", False),
            description="Fit Intercept")
        setattr(widget_fitIntercept, 'name', 'fitIntercept')

        widget_labelCol = widgets.Text(
            value=dict.get("labelCol", "label"),
            description="Label Column")
        setattr(widget_labelCol, 'name', 'labelCol')

        widget_maxIter = widgets.IntRangeSlider(
            value=(dict.get("maxIter", 100), 150),
            min=10,
            max=200,
            step=1,
            description="Max iterations")
        setattr(widget_maxIter, 'name', 'maxIter')


        widgets_naive_bayes = ParamsClassification.create_naive_bayes_widgets(dict)[:-1] # we don't need the last one

        all_lists = [[widget_elasticNetParam, widget_regParam, widget_maxIter],
                     [widget_fitIntercept, widget_threshold, widget_tol]]

        return list(map(lambda x: widgets.HBox(x), all_lists)) + widgets_naive_bayes

    @staticmethod
    def create_decision_tree_classifier_widgets(dict):
        '''
        instiantate the widgets for decision tree classifier
        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        '''

        widget_cacheNodeIds = widgets.Checkbox(
            value=dict.get("cacheNodeIds", False),
            description="Cache Node Id")
        setattr(widget_cacheNodeIds, 'name', 'cacheNodeIds')

        widget_checkpointInterval = widgets.IntRangeSlider(
            value=(1, dict.get("checkpointInterval", 10)),
            min=1,
            max=100,
            step=5,
            description="Check point interval")
        setattr(widget_checkpointInterval, 'name', "checkpointInterval")

        widget_impurity = widgets.Select(
            value=dict.get("impurity", "gini"),
            options=["gini", "entropy"],
            description="Impurity")
        setattr(widget_impurity, 'name', "impurity")

        widget_maxBins = widgets.IntRangeSlider(
            value=(16, dict.get("maxBins", 32)),
            min=2,
            max=256,
            step=2,
            description="Maximum bins")
        setattr(widget_maxBins, 'name', "maxBins")

        widget_maxDepth = widgets.IntRangeSlider(
            value=(3, dict.get("maxDepth", 5)),
            min=1,
            max=50,
            step=1,
            description="Maximum Depth")
        setattr(widget_maxDepth, 'name', "maxDepth")

        widget_maxMemoryInMB = widgets.IntRangeSlider(
            value=(128, dict.get("maxMemoryInMB", 256)),
            min=8,
            max=1024,
            step=8,
            description="Maximum Memory MB")
        setattr(widget_maxMemoryInMB, 'name', "maxMemoryInMB")

        widget_minInfoGain = widgets.FloatRangeSlider(
            value=(dict.get("minInfoGain", 0.0), 0.5),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Minimum Information Gain")
        setattr(widget_minInfoGain, 'name', "minInfoGain")

        widget_minInstancesPerNode = widgets.IntRangeSlider(
            value=(dict.get("minInstancesPerNode", 1), 2),
            min=1,
            max=10,
            step=1,
            description="Minimum number of instances pr node")
        setattr(widget_minInstancesPerNode, 'name', "minInstancesPerNode")

        list_widgets_all = [[widget_checkpointInterval, widget_maxBins, widget_maxDepth],
                            [widget_maxMemoryInMB, widget_minInfoGain, widget_minInstancesPerNode],
                            [widget_impurity, widget_cacheNodeIds]]
        widgets_decision_tree_classifier = list(map(lambda x: widgets.HBox(x), list_widgets_all))
        widgets_naive_bayes = ParamsClassification.create_naive_bayes_widgets(dict)[:-1]  # we don't need the last one

        return widgets_decision_tree_classifier + widgets_naive_bayes

    @staticmethod
    def create_GB_tree_classifier_widgets(dict):
        '''
        instiante the Gradient Boosted Tree classifier widgets
        :param dict: dictonary with widget names and default values
        :return: list with HBox's of widgets
        '''

        widget_stepSize = widgets.FloatRangeSlider(
            value=(0.0, dict.get("stepSize", 0.1)),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Step size")
        setattr(widget_stepSize, 'name', 'stepSize')

        widget_lossType = widgets.Select(
            value=dict.get("lossType", "logistic"),
            options=["logistic"],
            description="Loss type")
        setattr(widget_lossType, 'name', "lossType")

        widget_cacheNodeIds = widgets.Checkbox(
            value=dict.get("cacheNodeIds", False),
            description="Cache node id")
        setattr(widget_cacheNodeIds, 'name', 'cacheNodeIds')

        list_widgets_gbt = [widgets.HBox([widget_lossType, widget_stepSize,  widget_cacheNodeIds])]
        widgets_decision_tree_classifier = ParamsClassification.create_decision_tree_classifier_widgets(dict)
        del widgets_decision_tree_classifier[2]
        del widgets_decision_tree_classifier[-1]

        return list_widgets_gbt + widgets_decision_tree_classifier

    @staticmethod
    def create_random_forrest_classifier_widgets(dict):
        '''
        Initiate the widgets for random forest classifier
        :param dict: dictionary with _parameters and values
        :return: list with HBox's of widgets
        '''

        widget_featureSubsetStrategy = widgets.Select(
            value=dict.get("featureSubsetStrategy", 'auto'),
            options=['auto', "onethird", "sqrt", "log2", "(0.0-1.0]", "[1-n]", "all"],
            description="Feature subset strategy")
        setattr(widget_featureSubsetStrategy, 'name', "featureSubsetStrategy")

        widget_numTrees = widgets.IntRangeSlider(
            value=(dict.get("numTrees", 20), 40),
            min=1,
            max=100,
            step=1,
            description="Number of trees")
        setattr(widget_numTrees, 'name', "numTrees")

        widgets_decision_tree_classifier = ParamsClassification.create_decision_tree_classifier_widgets(dict)
        widgets_random_forrest_classifier = [widgets.HBox([widget_featureSubsetStrategy, widget_numTrees])]

        return widgets_decision_tree_classifier + widgets_random_forrest_classifier


    @staticmethod
    def create_naive_bayes_widgets(dict):
        '''
        Initiate the widgets for naive bayes
        :param dict: dictionary with parameter names and default values
        :return: list with Hbox's of widgets
        '''

        widget_modelType = widgets.Select(
            value=dict.get("modelType", "multinomial"),
            options=['multinomial', 'bernoulli'],
            description="Model type")
        setattr(widget_modelType, 'name', 'modelType')

        widget_probabilityCol = widgets.Text(
            value=dict.get("probabilityCol", "prbability"),
            description="Probability Column")
        setattr(widget_probabilityCol, 'name', 'probabilityCol')

        widget_rawPredictionCol = widgets.Text(
            value=dict.get("rawPredictionCol", "rawPrediction"),
            description="Raw prediction")
        setattr(widget_rawPredictionCol, 'name', 'rawPredictionCol')

        widget_smoothing = widgets.FloatRangeSlider(
            value=(0.5, dict.get("smoothing", 1.0)),
            min=0.0,
            max=2.0,
            step=0.01,
            description="Smoothing")
        setattr(widget_smoothing, 'name', 'smoothing')

        widgets_one_vs_rest = ParamsClassification.create_one_vs_rest(dict)

        return widgets_one_vs_rest + \
               [widgets.HBox([widget_probabilityCol, widget_rawPredictionCol]),
                widgets.HBox([widget_modelType, widget_smoothing])]

    @staticmethod
    def create_multi_layer_perception_widget(dict):

        '''
        Initiate widgets for multi layer perception _parameters.
        :param dict: dictionary with parameter names and default values
        :return: list with Hbox's with widgets
        '''

        widget_maxIter = widgets.IntRangeSlider(
            value=(50, dict.get("maxIter", 100)),
            min=10,
            max=200,
            step=1,
            description="Max iterations")
        setattr(widget_maxIter, 'name', 'maxIter')

        widget_solver = widgets.Select(
            value=dict.get("solver", "l-bfgs"),
            options=['l-bfgs', 'gd'],
            description="sovler")
        setattr(widget_solver, 'name', 'solver')

        widget_tol = widgets.FloatRangeSlider(
            value=(dict.get("tol", 1e-04), 1e-03),
            min=1e-04,
            max=1e-01,
            step=1e-01,
            description="Tolerance")
        setattr(widget_tol,'name', "tol")

        widget_stepSize = widgets.FloatRangeSlider(
            value=(dict.get("stepSize", 0.1), 0.3),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Step size",
            name="stepSize")
        setattr(widget_stepSize, 'name', 'stepSize')

        widget_blockSize = widgets.IntRangeSlider(
            value=(64, dict.get("blockSize", 128)),
            min=10,
            max=1000,
            step=1,
            description="Block size")
        setattr(widget_blockSize, 'name', 'blockSize')

        widgets_one_vs_rest = ParamsClassification.create_one_vs_rest(dict)

        return [widgets.HBox([widget_solver, widget_tol, widget_stepSize]),
                widgets.HBox([widget_blockSize, widget_maxIter])] + widgets_one_vs_rest

    @staticmethod
    def create_one_vs_rest(dict):

        widget_labelCol = widgets.Text(
            value=dict.get("labelCol", "label"),
            description="Label Column")
        setattr(widget_labelCol, 'name', "labelCol")

        widget_predictionCol = widgets.Text(
            value=dict.get("predictionCol", "prediction"),
            description="Prediction Column")
        setattr(widget_predictionCol, 'name', 'predictionCol')

        return [widgets.HBox([widget_labelCol, widget_predictionCol])]