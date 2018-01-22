from shared.parse_algorithm_variables import parse_algorithm_variables
from pyspark.sql import functions as F
# from examples import SemisupervisedMnist
from pyspark.sql import SparkSession
from pathlib import Path
from shared.Experiments import Experiments
import warnings
import functools
import numpy as np
from semisupervised.labelpropagation import label_propagation

default_lp_param = {'sigma': 340, 'tol':0.01, 'k': 10, 'max_iters': 5,
                    'eval_type': None, 'standardize': True, 'priors': None}

def run(sc, **kwargs):

    # select import data
    spark = SparkSession(sparkContext=sc)
    str_input_data = kwargs.get('input_data', None)
    if not Path(str_input_data).is_file():
        # file exists
        warnings.warn('Input file is not applicable using default, ', ImportWarning)
        str_input_data = '/home/svanhmic/workspace/data/DABAI/sparkdata/csv/double_helix3.csv'

    algo_types = parse_algorithm_variables(kwargs.get('algo_params', {}))
    for key in default_lp_param.keys():
        if key not in algo_types:
            algo_types[key] = default_lp_param[key]

    input_data_frame = spark.read.csv(
        path=str_input_data, header=True, inferSchema=True,
        mode='PERMISSIVE', nullValue=float('NAN'), nanValue=float('NAN'))

    list_label_id_cols = [kwargs.get('labels', None)] + [kwargs.get('id', None)]
    if kwargs.get('features', None):
        list_input_cols = kwargs.get('features', None)
    else:
        list_input_cols = [i for i in input_data_frame.columns if i not in list_label_id_cols]

    lp = functools.partial(
        label_propagation, id_col='id',
        label_col=list_label_id_cols[0], feature_cols=list_input_cols, **algo_types)

    keys = dict(filter(lambda x: x[0] not in ('sc'), lp.keywords.items()))

    ex = Experiments(data_size=[100, 1000, 10000, 100000])
    output_data_frame = ex.run_experiment(
        sc=sc, data=input_data_frame, functions=lp, known_fraction=0.1, **keys)

    # times, output_data_frame = label_propagation()
    return output_data_frame
