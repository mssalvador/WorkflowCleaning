from semisupervised.LabelPropagation import label_propagation
from shared.WorkflowLogger import logger_info_decorator, logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T

@logger_info_decorator
def run(sc, **kwargs):

    # set some stuff
    spark = SparkSession(sparkContext=sc)
    spark.conf.set("spark.sql.crossJoin.enabled", "true")
    input_data = kwargs.get('input', None)
    feature_columns = [T.StructField(f, T.DoubleType(), False) for f in kwargs.get('features', None)]
    label_columns = [T.StructField(kwargs.get('labels', None)[0], T.IntegerType(), True)]
    id_column = [T.StructField(kwargs.get('id', 'id'), T.IntegerType(), False)]
    algo_types = kwargs.get('algo_params', None)

    #Import data
    input_data_frame = spark.read.load(
        path=input_data, format='csv', schema=id_column+label_columns+feature_columns)

    # Execute algorithm
    output_data_frame = label_propagation(
        sc=sc, data_frame=input_data_frame, label_col=kwargs.get('labels', None),
        id_col=kwargs.get('id', 'id'), feature_cols=kwargs.get('features', None),
        k=algo_types['k'], sigma=algo_types['sigma'], max_iters=algo_types['max_iter'],
        tol=algo_types['tol'], standardize=algo_types['standardizer'],
        eval_type=algo_types['eval'], priors=algo_types['prior'])

    # Return result
    return output_data_frame