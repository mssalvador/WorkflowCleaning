import pyspark
import pyspark.sql.types as T
from shared.WorkflowLogger import logger_info_decorator, logger
from ast import literal_eval
from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow


@logger_info_decorator
def run(sc: pyspark.SparkContext, **kwargs):

    # Initialization phase v.1.0
    import_path = kwargs.get('input_data', None)
    feature_columns = kwargs.get('features', None)
    label_columns = kwargs.get('labels', 'k')
    id_column = kwargs.get('id', 'id')
    algorithm_params = _parse_algorithm_variables(kwargs.get('algo_params', None))
    standardizer = algorithm_params.get('standardizer', False)

    spark_session = pyspark.sql.SparkSession(sc)
    label_schema = [T.StructField(l, T.StringType(), False) for l in label_columns]
    id_schema = [T.StructField(idx, T.StringType(), False) for idx in id_column]
    feature_schema = [T.StructField(f, T.DoubleType(), False) for f in feature_columns]
    training_data_schema = T.StructType(id_schema+label_schema+feature_schema)
    training_data_frame = spark_session.read.load(
        path=import_path, format='csv', schema= training_data_schema)

    cleaning_workflow = ExecuteWorkflow(
        dict_params=algorithm_params, cols_features=feature_columns,
        cols_labels=label_columns,standardize=standardizer
    )

    training_model = cleaning_workflow.execute_pipeline(training_data_frame)
    clustered_data_frame = cleaning_workflow.apply_model(
        sc=sc, model=training_model, data_frame=training_data_frame)

    return clustered_data_frame


@logger_info_decorator
def _parse_algorithm_variables(vars):
    for key, val in vars.items():
        try:
            vars[key] = literal_eval(val)
        except ValueError as ve:
            print('Data {} is of type {}'.format(key, type(val)))
            logger.info('Data {} is of type {}'.format(key, type(val)))
        except SyntaxError as se:
            vars[key] = val.strip(' ')
            logger.error('Data {} is of type {}'.format(key, type(val)))
    return vars
