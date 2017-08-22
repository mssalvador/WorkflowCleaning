'''
Created on Jul 5, 2017

@author: svanhmic

@description: Create a widget page in a class, for selecting parameters for classification
'''


from ipywidgets import widgets
from traitlets import dlink
from pyspark.ml import classification
import logging
from pyspark import SparkContext
from shared import OwnFloatRangeSlider, OwnCheckBox, OwnIntRangeSlider, OwnSelect, OwnText, OwnDropdown

# setup logging first, better than print!
logger_parameter_select = logging.getLogger(__name__)
logger_parameter_select.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_classification.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_parameter_select.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)

sc = SparkContext.getOrCreate()


class ParamsClassification(object):

    algorithm_classification = [str(i) for i in classification.__all__
                                if ("Model" not in i) if ("Summary" not in i)]

    def __init__(self):
        logger_parameter_select.info(" Create_Classification_Parameters created")

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
            logger_parameter_select.debug(" Parameters selected for algorithm {} with _parameters {}".format(i, algo_and_params[i]))
        return algo_and_params

    def select_parameters(self):

        '''
        The main method for selecting _parameters to each algorithm. Each algorithm has its own set of _parameters.
        :return: None
        '''
        widget_dropdown_algorithms = OwnDropdown.OwnDropdown(options=ParamsClassification.algorithm_classification,
                                                             value=ParamsClassification.algorithm_classification[0],
                                                             description="Algorithms",
                                                             disabled=False,
                                                             name="algorithm")

        all_widgets = widgets.VBox([widget_dropdown_algorithms])

        list_of_methods = [self.create_logistic_regression_widgets,
                           self.create_decision_tree_classifier_widgets,
                           self.create_GB_tree_classifier_widgets,
                           self.create_random_forrest_classifier_widgets,
                           self.create_naive_bayes_widgets,
                           self.create_multi_layer_perception_widget,
                           self.create_one_vs_rest]

        widget_and_algorithms = dict(zip(ParamsClassification.algorithm_classification, list_of_methods))

        def update_algorithm_parameters(change):

            if change in widget_and_algorithms.keys():
                logger_parameter_select.debug(" Algorithm changed to: {}".format(change))
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
        widget_elasticNetParam = OwnFloatRangeSlider.OwnFloatRangeSlider(value= (dict.get("elasticNetParam", 0.0), 0.5),
                                                                         min=0.0,
                                                                         max=1.0,
                                                                         step=0.01,
                                                                         description="Elastic Net",
                                                                         name="elasticNetParam")

        widget_fitIntercept = OwnCheckBox.OwnCheckBox(value=dict.get("fitIntercept", False),
                                                      description="Fit Intercept",
                                                      name="fitIntercept")

        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_maxIter = OwnIntRangeSlider.OwnIntRangeSlider(value=(dict.get("maxIter", 100), 150),
                                                             min=10,
                                                             max=200,
                                                             step=1,
                                                             description="Max iterations",
                                                             name="maxIter")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_probabilityCol = OwnText.OwnText(value=dict.get("probabilityCol", "probability"),
                                                description="Probability Column",
                                                name="probabilityCol")

        widget_rawPredictionCol = OwnText.OwnText(value=dict.get("rawPredictionCol", "rawPrediction"),
                                                  description="Raw Prediction Column",
                                                  name="rawPredictionCol")

        widget_regParam = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("regParam", 0.0), 0.5),
                                                                  min=0.0,
                                                                  max=1.0,
                                                                  step=0.01,
                                                                  description="Regularization",
                                                                  name="regParam")

        widget_threshold = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(0.0, dict.get("threshold", 0.5)),
                                                                   min=0.0,
                                                                   max=1.0,
                                                                   step=0.01,
                                                                   description="Threshold",
                                                                   name="threshold")

        widget_tol = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("tol", 0.001), 0.01),
                                                             min=0.01,
                                                             max=0.1,
                                                             step=0.01,
                                                             description="Tolerance",
                                                             name="tol")

        all_lists = [[widget_elasticNetParam, widget_regParam, widget_maxIter],
                     [widget_predictionCol, widget_probabilityCol],
                     [widget_rawPredictionCol, widget_labelCol],
                     [widget_fitIntercept, widget_threshold, widget_tol]]

        return list(map(lambda x: widgets.HBox(x), all_lists))

    @staticmethod
    def create_decision_tree_classifier_widgets(dict):
        '''
        instiantate the widgets for decision tree classifier
        :param dict: name of _parameters and its default value
        :return: list with HBox's of widgets
        '''

        widget_cacheNodeIds = OwnCheckBox.OwnCheckBox(value=dict.get("cacheNodeIds", False),
                                                      description="Cache Node Id",
                                                      name="cacheNodeIds")

        widget_checkpointInterval = OwnIntRangeSlider.OwnIntRangeSlider(value=(1, dict.get("checkpointInterval", 10)),
                                                                        min=1,
                                                                        max=100,
                                                                        step=5,
                                                                        description="Check point interval",
                                                                        name="checkpointInterval")

        widget_impurity = OwnSelect.OwnSelect(value=dict.get("impurity", "gini"),
                                              options=["gini", "entropy"],
                                              description="Impurity",
                                              name="impurity")

        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_maxBins = OwnIntRangeSlider.OwnIntRangeSlider(value=(16, dict.get("maxBins", 32)),
                                                             min=2,
                                                             max=256,
                                                             step=2,
                                                             description="Maximum bins",
                                                             name="maxBins")

        widget_maxDepth = OwnIntRangeSlider.OwnIntRangeSlider(value=(3, dict.get("maxDepth", 5)),
                                                              min=1,
                                                              max=50,
                                                              step=1,
                                                              description="Maximum Depth",
                                                              name="maxDepth")

        widget_maxMemoryInMB = OwnIntRangeSlider.OwnIntRangeSlider(value=(128, dict.get("maxMemoryInMB", 256)),
                                                                   min=8,
                                                                   max=1024,
                                                                   step=8,
                                                                   description="Maximum Memory MB",
                                                                   name="maxMemoryInMB")

        widget_minInfoGain = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("minInfoGain", 0.0), 0.5),
                                                                     min=0.0,
                                                                     max=1.0,
                                                                     step=0.01,
                                                                     description="Minimum Information Gain",
                                                                     name="minInfoGain")

        widget_minInstancesPerNode = OwnIntRangeSlider.OwnIntRangeSlider(value=(dict.get("minInstancesPerNode", 1), 2),
                                                                         min=1,
                                                                         max=10,
                                                                         step=1,
                                                                         description="Minimum number of instances pr node",
                                                                         name="minInstancesPerNode")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_probabilityCol = OwnText.OwnText(value=dict.get("probabilityCol", "prbability"),
                                                description="Probability Column",
                                                name="probabilityCol")

        widget_rawPredictionCol = OwnText.OwnText(value=dict.get("rawPredictionCol", "rawPrediction"),
                                                  description="Raw prediction",
                                                  name="rawPredictionCol")

        return [widgets.HBox([widget_predictionCol, widget_probabilityCol, widget_rawPredictionCol]),
                widgets.HBox([widget_labelCol, widget_impurity, widget_cacheNodeIds]),
                widgets.HBox([widget_checkpointInterval, widget_maxBins, widget_maxDepth]),
                widgets.HBox([widget_maxMemoryInMB, widget_minInfoGain, widget_minInstancesPerNode])]

    @staticmethod
    def create_GB_tree_classifier_widgets(dict):
        '''
        instiante the Gradient Boosted Tree classifier widgets
        :param dict: dictonary with widget names and default values
        :return: list with HBox's of widgets
        '''
        widget_cacheNodeIds = OwnCheckBox.OwnCheckBox(value=dict.get("cacheNodeIds", False),
                                                      description="Cache node id",
                                                      name="cacheNodeIds")

        widget_checkpointInterval = OwnIntRangeSlider.OwnIntRangeSlider(value=(5, dict.get("checkpointInterval", 10)),
                                                                        min=1,
                                                                        max=100,
                                                                        step=5,
                                                                        description="Check point interval",
                                                                        name="checkpointInterval")

        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_lossType = OwnSelect.OwnSelect(value=dict.get("lossType", "logistic"),
                                              options=["logistic"],
                                              description="Loss type",
                                              name="lossType")

        widget_maxBins = OwnIntRangeSlider.OwnIntRangeSlider(value=(16, dict.get("maxBins", 32)),
                                                             min=2,
                                                             max=256,
                                                             step=2,
                                                             description="Maximum bins",
                                                             name="maxBins")

        widget_maxDepth = OwnIntRangeSlider.OwnIntRangeSlider(value=(3, dict.get("maxDepth", 5)),
                                                              min=1,
                                                              max=50,
                                                              step=1,
                                                              description="Maximum Depth",
                                                              name="maxDepth")

        widget_maxMemoryInMB = OwnIntRangeSlider.OwnIntRangeSlider(value=(128, dict.get("maxMemoryInMB", 256)),
                                                                   min=8,
                                                                   max=1024,
                                                                   step=8,
                                                                   description="Maximum Memory MB:",
                                                                   name="maxMemoryInMB")

        widget_minInfoGain = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("minInfoGain", 0.0), 0.3),
                                                                     min=0.0,
                                                                     max=1.0,
                                                                     step=0.01,
                                                                     description="Minimum Information Gain",
                                                                     name="minInfoGain")

        widget_minInstancesPerNode = OwnIntRangeSlider.OwnIntRangeSlider(value=(dict.get("minInstancesPerNode", 1), 2),
                                                                         min=1,
                                                                         max=10,
                                                                         step=1,
                                                                         description="Minimum number of instances pr. node",
                                                                         name="minInstancesPerNode")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_stepSize = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(0.0, dict.get("stepSize", 0.1)),
                                                                  min=0.0,
                                                                  max=1.0,
                                                                  step=0.01,
                                                                  description="Step size",
                                                                  name="stepSize")

        return [widgets.HBox([widget_stepSize, widget_minInfoGain, widget_cacheNodeIds]),
                widgets.HBox([widget_checkpointInterval, widget_maxBins, widget_maxDepth]),
                widgets.HBox([widget_maxMemoryInMB, widget_minInstancesPerNode, widget_lossType]),
                widgets.HBox([widget_predictionCol, widget_labelCol])]

    @staticmethod
    def create_random_forrest_classifier_widgets(dict):
        '''
        Initiate the widgets for random forest classifier
        :param dict: dictionary with _parameters and values
        :return: list with HBox's of widgets
        '''
        widget_cacheNodeIds = OwnCheckBox.OwnCheckBox(value=dict.get("cacheNodeIds", False),
                                                      description="Cache Node Id",
                                                      name="cacheNodeIds")

        widget_checkpointInterval = OwnIntRangeSlider.OwnIntRangeSlider(value=(5, dict.get("checkpointInterval", 10)),
                                                                        min=1,
                                                                        max=100,
                                                                        step=5,
                                                                        description="Check point interval",
                                                                        name="checkpointInterval")

        widget_featureSubsetStrategy = OwnSelect.OwnSelect(value=dict.get("featureSubsetStrategy", 'auto'),
                                                           options=['auto', "onethird", "sqrt", "log2", "(0.0-1.0]", "[1-n]", "all"],
                                                           description="Feature subset strategy",
                                                           name="featureSubsetStrategy")

        widget_impurity = OwnSelect.OwnSelect(value=dict.get("impurity", "gini"),
                                              options=["gini", "entropy"],
                                              description="Impurity",
                                              name="impurity")

        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_maxBins = OwnIntRangeSlider.OwnIntRangeSlider(value=(16, dict.get("maxBins", 32)),
                                                             min=2,
                                                             max=256,
                                                             step=2,
                                                             description="Maximum bins",
                                                             name="maxBins")

        widget_maxDepth = OwnIntRangeSlider.OwnIntRangeSlider(value=(3, dict.get("maxDepth", 5)),
                                                              min=1,
                                                              max=50,
                                                              step=1,
                                                              description="Maximum Depth",
                                                              name="maxDepth")

        widget_maxMemoryInMB = OwnIntRangeSlider.OwnIntRangeSlider(value=(128, dict.get("maxMemoryInMB", 256)),
                                                                   min=8,
                                                                   max=1024,
                                                                   step=8,
                                                                   description="Maximum Memory MB:",
                                                                   name="maxMemoryInMB")

        widget_minInfoGain = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("minInfoGain", 0.0), 0.2),
                                                                     min=0.0,
                                                                     max=1.0,
                                                                     step=0.01,
                                                                     description="Minimum Information Gain",
                                                                     name="minInfoGain")

        widget_minInstancesPerNode = OwnIntRangeSlider.OwnIntRangeSlider(value=(dict.get("minInstancesPerNode", 1), 3),
                                                                         min=1,
                                                                         max=10,
                                                                         step=1,
                                                                         description="Minimum number of instances pr. node",
                                                                         name="minInstancesPerNode")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_probabilityCol = OwnText.OwnText(value=dict.get("probabilityCol", "prbability"),
                                                description="Probability Column",
                                                name="probabilityCol")

        widget_rawPredictionCol = OwnText.OwnText(value=dict.get("rawPredictionCol", "rawPrediction"),
                                                  description="Raw prediction",
                                                  name="rawPredictionCol")

        widget_numTrees = OwnIntRangeSlider.OwnIntRangeSlider(value=(5, dict.get("numTrees", 20), 40),
                                                              min=1,
                                                              max=100,
                                                              step=1,
                                                              description="Number of trees",
                                                              name="numTrees")

        return [widgets.HBox([widget_featureSubsetStrategy, widget_impurity, widget_cacheNodeIds]),
                widgets.HBox([widget_checkpointInterval, widget_maxBins, widget_maxDepth]),
                widgets.HBox([widget_maxMemoryInMB, widget_minInfoGain]),
                widgets.HBox([widget_numTrees, widget_minInstancesPerNode]),
                widgets.HBox([widget_predictionCol, widget_probabilityCol]),
                widgets.HBox([widget_rawPredictionCol, widget_labelCol])]

    @staticmethod
    def create_naive_bayes_widgets(dict):
        '''
        Initiate the widgets for naive bayes
        :param dict: dictionary with parameter names and default values
        :return: list with Hbox's of widgets
        '''
        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_modelType = OwnSelect.OwnSelect(value=dict.get("modelType", "multinomial"),
                                               options=['multinomial', 'bernoulli'],
                                               description="Model type",
                                               name="modelType")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_probabilityCol = OwnText.OwnText(value=dict.get("probabilityCol", "prbability"),
                                                description="Probability Column",
                                                name="probabilityCol")

        widget_rawPredictionCol = OwnText.OwnText(value=dict.get("rawPredictionCol", "rawPrediction"),
                                                  description="Raw prediction",
                                                  name="rawPredictionCol")

        widget_smoothing = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(0.5, dict.get("smoothing", 1.0)),
                                                                   min=0.0,
                                                                   max=2.0,
                                                                   step=0.01,
                                                                   description="Smoothing",
                                                                   name="smoothing")

        return [widgets.HBox([widget_modelType, widget_smoothing]),
                widgets.HBox([widget_labelCol, widget_predictionCol]),
                widgets.HBox([widget_probabilityCol, widget_rawPredictionCol])]

    @staticmethod
    def create_multi_layer_perception_widget(dict):

        '''
        Initiate widgets for multi layer perception _parameters.
        :param dict: dictionary with parameter names and default values
        :return: list with Hbox's with widgets
        '''
        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_maxIter = OwnIntRangeSlider.OwnIntRangeSlider(value=(50, dict.get("maxIter", 100)),
                                                             min=10,
                                                             max=200,
                                                             step=1,
                                                             description="Max iterations",
                                                             name="maxIter")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        widget_solver = OwnSelect.OwnSelect(value=dict.get("solver", "l-bfgs"),
                                            options=['l-bfgs', 'gd'],
                                            description="sovler",
                                            name="solver")

        widget_tol = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("tol", 1e-04), 1e-03),
                                                             min=1e-04,
                                                             max=1e-01,
                                                             step=1e-01,
                                                             description="Tolerance",
                                                             name="tol")

        widget_stepSize = OwnFloatRangeSlider.OwnFloatRangeSlider(value=(dict.get("stepSize", 0.1), 0.3),
                                                                  min=0.0,
                                                                  max=1.0,
                                                                  step=0.01,
                                                                  description="Step size",
                                                                  name="stepSize")

        widget_blockSize = OwnIntRangeSlider.OwnIntRangeSlider(value=(64, dict.get("blockSize", 128)),
                                                               min=10,
                                                               max=1000,
                                                               step=1,
                                                               description="Block size",
                                                               name="blockSize")

        return [widgets.HBox([widget_solver, widget_tol, widget_stepSize]),
                widgets.HBox([widget_blockSize, widget_maxIter]),
                widgets.HBox([widget_predictionCol, widget_labelCol])]

    @staticmethod
    def create_one_vs_rest(dict):

        widget_labelCol = OwnText.OwnText(value=dict.get("labelCol", "label"),
                                          description="Label Column",
                                          name="labelCol")

        widget_predictionCol = OwnText.OwnText(value=dict.get("predictionCol", "prediction"),
                                               description="Prediction Column",
                                               name="predictionCol")

        return [widgets.HBox([widget_labelCol]), widgets.HBox([widget_predictionCol])]