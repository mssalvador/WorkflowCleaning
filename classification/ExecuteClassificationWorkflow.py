'''
Created on Jul 10, 2017

@author: svanhmic

@description: Create a widget page in a class, for selecting parameters for classification
'''

import logging
import sys

logger_execute = logging.getLogger(__name__)
logger_execute.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_classification.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_execute.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)


class ExecuteWorkflowClassification(object):

    def __init__(self, dict_params=None, standardize=False, featureCols=None, labelCols=None):
        '''
        Constructor for ExecuteWorkflowClassification.
        :param dict_params:
        :param standardize:
        :param featureCols:
        :param labelCols:
        '''

        try:
            self._algorithm = dict_params.pop('algorithm', 'LogisticRegression')
        except AttributeError as ae:
            tb = sys.exc_info()[2]
            logger_execute.warning(ae.with_traceback(tb))
            self._algorithm = 'LogisticRegression'

        self._params = dict_params
        self._standardize = standardize
        self._featureCols = featureCols
        self._labelCols = labelCols

        try:
            self._pipeline, self._parameter_grid = self.create_standard_pipeline()
        except TypeError as te:
            tb = sys.exc_info()[2]
            logger_execute.warning(te.with_traceback(tb))
        logger_execute.info("ExecuteWorkflowClassification created with '{}'"
                            .format(str(ExecuteWorkflowClassification)))

    @property
    def parameter_grid(self):
        logger_execute.debug('Exported parameter_grid: {}'
                             .format(self._parameter_grid))
        return self._parameter_grid

    @property
    def pipeline(self):
        logger_execute.debug('Exported pipeline: {}'
                             .format(self._pipeline.getStages()))
        return self._pipeline

    def __str__(self):
        return "algorithm: {}".format(self._algorithm)

    def __repr__(self):
        return "ExecuteWorkflowClassification('{}')".format(self._params)

    def create_custom_pipeline(self):

        '''
        TODO: create a method that can create a custom transformation... This could be a good opportunity
        :return:
        '''

        # import statements
        from pyspark.ml import feature
        from ipywidgets import widgets

        # show-off
        all_transformation = filter(lambda x: "Model" not in x, feature.__all__)
        print("\n".join(all_transformation))

    def create_standard_pipeline(self):
        '''
        This method creates a standard pipeline, standard meaning: vectorize, standardize and model...
        :return: Pipeline for pyspark, ParameterGrid for Pyspark pipeline
        '''

        # Import statements
        from pyspark.ml import classification
        from pyspark.ml import Pipeline, tuning
        from pyspark.ml import feature as Feat

        # Feature columns are created from instance variables
        feature_columns = [i.name for i in self._featureCols]

        # Vectorized transformation
        vectorizer = Feat.VectorAssembler(
            inputCols=feature_columns,
            outputCol="vectorized_features")

        # Standardize estimator
        if self._standardize:
            standardizes = Feat.StandardScaler(
                withMean=True,
                withStd=True,
                inputCol=vectorizer.getOutputCol(),
                outputCol="scaled")
        else:
            standardizes = Feat.StandardScaler(
                withMean=False,
                withStd=False,
                inputCol=vectorizer.getOutputCol(),
                outputCol="scaled")

        # Labels and strings are already set into the model, +
        label_dict = dict(filter(lambda x: not isinstance(x[1], tuple), self._params.items()))
        feature_dict = dict(ExecuteWorkflowClassification.generate_equidistant_params(self._params))
        label_dict['featuresCol'] = standardizes.getOutputCol()
        #print(label_dict)

        # Model is set
        model = eval("classification." + self._algorithm)(**label_dict)

        # Parameter is set
        param_grid = tuning.ParamGridBuilder()
        for idx, val in feature_dict.items():
            param_grid.addGrid(eval('model.'+idx), val)

        pipe = Pipeline(stages=[vectorizer, standardizes, model])
        return pipe, param_grid.build()

    def run_cross_val(self, data, evaluator, n_folds):
        '''
        :param data:
        :param evaluator:
        :param n_folds:
        :return:
        '''

        from pyspark.ml.tuning import CrossValidator
        from pyspark.ml import evaluation

        if not isinstance(evaluator, evaluation.Evaluator):
            print("this {} is not good. Should have been of type {}".format(evaluator, evaluation.Evaluator))
            logger_execute.warning('evaluator invalid. Cannot use: {}'.format(evaluator))
            return

        #print(self._parameter_grid)
        cv = CrossValidator(
            estimator=self._pipeline,
            estimatorParamMaps=self._parameter_grid,
            evaluator=evaluator,
            numFolds=n_folds)

        return cv.fit(data)

    @staticmethod
    def generate_equidistant_params(dict_param, number_of_spaces=3):
        '''
        Create a generator for parameter list
        :param dict_param:
        :param number_of_spaces:
        :return:
        '''

        import numpy as np
        params = dict(filter(lambda x: isinstance(x[1], tuple), dict_param.items()))
        for key, val in params.items():
            if isinstance(val, tuple) and isinstance(val[0], int):
                yield (key, np.linspace(val[0], val[1], number_of_spaces, True, dtype=int))
            else:
                yield (key, np.linspace(val[0], val[1], number_of_spaces, True, dtype=float))

