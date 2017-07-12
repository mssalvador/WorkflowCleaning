'''
Created on Jul 10, 2017

@author: svanhmic

@description: Create a widget page in a class, for selecting parameters for classification
'''


import logging

logger_execute = logging.getLogger(__name__)
logger_execute.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_classification.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_execute.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)

class ExecuteWorkflowClassification(object):

    def __init__(self, dict_params, standardize):

        self._params = dict_params
        self._standardize = standardize
        logger_execute.info("ExecuteWorkflowClassification created with '{}'"
                            .format(str(ExecuteWorkflowClassification)))

    @staticmethod
    def show_parameters(parameters):
        try:
            assert isinstance(parameters, dict)
            output_string = ', '.join(['{:s} = {}'
                                      .format(key, val) for key, val in parameters])
            print(output_string)

        except AssertionError:
            logger_execute.warning("Parameter '{}' is not of type dict but type '{}'"
                                   .format(parameters, type(parameters)))

    def __str__(self):
        return "algorithm: {}".format(self._params.get("algortihm", None))

    def __repr__(self):
        return "ExecuteWorkflowClassification('{}')".format(self._params)

    def create_pipeline(self):
        pass