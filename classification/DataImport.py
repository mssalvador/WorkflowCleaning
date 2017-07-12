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

from pyspark import SparkContext

