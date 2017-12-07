import pyspark
import pyspark.sql.types as T
from pyspark.ml import clustering
from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow

def run(sc : pyspark.SparkContext, **kwargs):

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
        training_model, training_data_frame)

    clustered_data_frame.head(5)
    return clustered_data_frame

def _parse_algorithm_variables(vars):
    lower_algos_dict = dict([
        (a.lower(), a) for a in clustering.__all__
        if ("Model" not in a) if ("Summary" not in a)
        if ("BisectingKMeans" not in a)])

    algorithm = lower_algos_dict[vars['algorithm'].lower()]
    model = getattr(clustering, algorithm)()
    param_map = [str(i.name).lower() for i in model.params]

    # Make sure that the params in self._params are the right for the algorithm
    params_labels = filter(lambda x: x[0].lower() in param_map, vars.items())
    return dict(params_labels)