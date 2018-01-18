from shared.parse_algorithm_variables import parse_algorithm_variables
from pyspark.sql import functions as F
from examples import SemisupervisedMnist
from pyspark.sql import SparkSession
from pathlib import Path
import warnings
import functools

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

    output_data_frame = SemisupervisedMnist.run_experiment(
        sc=sc, dataframe=input_data_frame, label_col=list_label_id_cols[0],
        feature_cols=list_input_cols, data_size=1000, fractions=0.1, **algo_types)

    output_data_frame = output_data_frame.withColumn(
        colName='error',
        col=F.when(F.col(list_label_id_cols[0]) == F.col('new_'+list_label_id_cols[0]), 0).otherwise(1))

    return output_data_frame.select(
        'id', list_label_id_cols[0],'new_'+list_label_id_cols[0], 'error', 'probabilities')