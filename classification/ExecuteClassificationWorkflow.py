"""
Created on Jul 10, 2017

@author: svanhmic

@description: Create a widget page in a class, for selecting parameters for classification
"""

import logging
import sys
from shared.WorkflowLogger import logger_info_decorator
# Import statements
from pyspark.ml import classification
from pyspark.ml import Pipeline, tuning
from pyspark.ml.feature import VectorAssembler, StandardScaler
from shared.ConvertAllToVecToMl import ConvertAllToVecToMl
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import evaluation
import numpy as np


class ExecuteWorkflowClassification(object):

    # @logger_info_decorator
    def __init__(self,
                 dict_params=None,
                 standardize=False,
                 feature_cols=None,
                 label_col=None,
                 id_col=None):
        """
        Constructor for ExecuteWorkflowClassification.
        :param dict_params:
        :param standardize:
        :param feature_cols:
        :param label_col:
        """
        self._algorithm = ExecuteWorkflowClassification._check_algorithm(
            algorithm=dict_params.pop('algorithm', 'LogisticRegression')
        )
        self._params = dict_params
        self._standardize = standardize
        self._feature_cols = feature_cols
        self._label_col = label_col
        self._id_col = id_col
        self._pipeline = None
        try:
            self._pipeline = self.create_standard_pipeline()
        except TypeError as te:
            tb = sys.exc_info()[2]
            print(te)
            print('something went wrong in pipeline')

    @property
    @logger_info_decorator
    def parameter_grid(self):
        return self._parameter_grid

    @property
    @logger_info_decorator
    def pipeline(self):
        return self._pipeline

    def __str__(self):
        return "algorithm: {}".format(self._algorithm)

    def __repr__(self):
        return "ExecuteWorkflowClassification('{}')".format(self._params)

    # @logger_info_decorator
    @staticmethod
    def _check_algorithm(algorithm='LogisticRegression'):
        try:
            applicable_algos = {'logisticregression': 'LogisticRegression'}
            return applicable_algos[algorithm.lower()]
        except AttributeError as ae:
            return 'LogisticRegression'

    def create_custom_pipeline(self):

        """
        TODO: create a method that can create a custom transformation... This could be a good opportunity
        :return:
        """

        # import statements
        from pyspark.ml import feature
        from ipywidgets import widgets

        # show-off
        all_transformation = filter(lambda x: "Model" not in x, feature.__all__)
        print("\n".join(all_transformation))

    def create_standard_pipeline(self, cross_validate=False):
        """
        This method creates a standard pipeline, standard meaning: vectorize, standardize and model...
        :return: Pipeline for pyspark, ParameterGrid for Pyspark pipeline
        """

        # Feature columns are created from instance variables
        # feature_columns = [i.name for i in self._feature_cols]

        # Vectorized transformation
        vectorizer = VectorAssembler(inputCols=self._feature_cols, outputCol='v_features')
        # Cast the vector from mllib to ml
        converter = ConvertAllToVecToMl(inputCol=vectorizer.getOutputCol(), outputCol='casted')
        # Standardize estimator
        standardizes = StandardScaler(
            withMean=self._standardize, withStd=self._standardize,
            inputCol=converter.getOutputCol(), outputCol="scaled"
        )
        # Labels and strings are already set into the model, +
        dict_parameters = dict(filter(lambda x: not isinstance(x[1], tuple), self._params.items()))
        dict_parameters['featuresCol'] = standardizes.getOutputCol()
        dict_parameters['labelCol'] = self._label_col[0] # HACK!!!
        #print(label_dict)

        # Model is set
        model = eval("classification." + self._algorithm)(**dict_parameters)

        pipe = Pipeline(stages=[vectorizer, converter, standardizes, model])
        return pipe

    def create_grid(self, dict_features):
        param_grid = tuning.ParamGridBuilder()
        for model_parameter, grid_values in dict_features.items():
            if isinstance(grid_values, int) or isinstance(grid_values, float):
                param_grid.baseOn(eval('model.' + model_parameter), grid_values)
            else:
                param_grid.addGrid(eval('model.' + model_parameter), grid_values)

    @logger_info_decorator
    def run_cross_val(self, data, evaluator, n_folds):
        """
        :param data:
        :param evaluator:
        :param n_folds:
        :return:
        """
        if not isinstance(evaluator, evaluation.Evaluator):
            print("this {} is not good. Should have been of type {}".format(evaluator, evaluation.Evaluator))
            return

        # print(self._parameter_grid)
        cv = CrossValidator(
            estimator=self._pipeline,
            estimatorParamMaps=self._parameter_grid,
            evaluator=evaluator,
            numFolds=n_folds)

        return cv.fit(data)

    @staticmethod
    def generate_equidistant_params(dict_param, number_of_spaces=3):
        """
        Create a generator for parameter list
        :param dict_param:
        :param number_of_spaces:
        :return:
        """
        params = dict(filter(lambda x: isinstance(x[1], tuple), dict_param.items()))
        for key, val in params.items():
            if isinstance(val, tuple) and isinstance(val[0], int):
                yield (key, np.linspace(val[0], val[1], number_of_spaces, True, dtype=int))
            else:
                yield (key, np.linspace(val[0], val[1], number_of_spaces, True, dtype=float))


