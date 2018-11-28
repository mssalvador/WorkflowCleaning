import time
import functools
from pyspark.sql import types as T
# from semisupervised.labelpropagation.lp2 import label_propagation2
from pyspark.sql import SparkSession
default_lp_param = {'sigma': 340, 'tol':0.01, 'k': 10, 'max_iters': 5,
                    'eval_type': None, 'standardize': True, 'priors': None}


def run(sc, **kwargs):
    from semisupervised.labelpropagation.label_propagation import label_propagation
    from shared.Experiments import Experiments
    from shared.parse_algorithm_variables import parse_algorithm_variables


    # select import data
    spark = SparkSession(sparkContext=sc)
    str_input_data = kwargs.get('input_data', None)
    # if not Path(str_input_data).is_file():
    #     # file exists
    #     warnings.warn('Input file is not applicable using default, ', ImportWarning)
    #     str_input_data = '/home/svanhmic/workspace/data/DABAI/sparkdata/csv/double_helix3.csv'

    algo_types = parse_algorithm_variables(kwargs.get('algo_params', {}))
    if kwargs.get('id','id') == None:
        id_col = 'id'
    else:
        id_col = kwargs.get('id','id')

    for key in default_lp_param.keys():
        if key not in algo_types:
            algo_types[key] = default_lp_param[key]

    input_data_frame = spark.read.csv(
        path=str_input_data, header=True,
        inferSchema=True, mode='PERMISSIVE',
        nullValue=float('NAN'), nanValue=float('NAN')
    )
    input_data_frame.persist()
    list_label_id_cols = [kwargs.get('labels', None)] + [id_col]
    if kwargs.get('features', None):
        list_input_cols = kwargs.get('features', None)
    else:
        list_input_cols = [i for i in input_data_frame.columns
                           if i not in list_label_id_cols]

    # lp2 = label_propagation2(sc=sc,id_col='id',label_col=list_label_id_cols[0],feature_cols=list_input_cols, **algo_types)
    lp = functools.partial(
        func=label_propagation, id_col=id_col,
        label_col=kwargs.get('labels', None),
        feature_cols=list_input_cols, **algo_types
    )

    # output_data_frame = lp2.run(input_data_frame)
    keys = dict(filter(
        function=lambda x: x[0] not in ('sc'),
        iterable=lp.keywords.items())
    )

    ex = Experiments(data_size=[1000])
    output_data_frame = ex.run_experiment(
        sc=sc, data=input_data_frame,
        functions=lp, known_fraction=0.1, **keys
    )
    # times, output_data_frame = label_propagation()
    return output_data_frame
