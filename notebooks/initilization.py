# Make sure that Python starts in Workflow-folder or else the modules will be screewed up!
import sys
import os
import getpass
import datetime

# Graphics importation and pandas
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sb.set(style='whitegrid')

# Something about adding a thing to Python path.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)

# What's the time?
today = datetime.date.today()

# Identify user
user = getpass.getuser()
if user == "sidsel":
    parquet_path = "/home/sidsel/workspace/sparkdata/parquet"
elif user == "svanhmic":
    parquet_path = "/home/svanhmic/workspace/data/DABAI/sparkdata/parquet"
    
# Start the logger.
import logging
logger_tester = logging.getLogger(__name__)
logger_tester.setLevel(logging.INFO)
logger_file_handler_param = logging.FileHandler(
    '/tmp/workflow_notebook_test_{!s}.log'.format(today.strftime('%d-%m-%Y')))
logger_formatter_param = logging.Formatter(
    '%(asctime)s;%(levelname)s;%(name)s;%(message)s')

logger_tester.addHandler(logger_file_handler_param)
logger_file_handler_param.setFormatter(logger_formatter_param)

