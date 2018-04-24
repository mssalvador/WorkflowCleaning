from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import tuning
from shared.parse_algorithm_variables import parse_algorithm_variables
from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification


def run(sc: SparkContext, **kwargs):
    """
    :param sc:
    :param kwargs:
    :return:
    """
    # Import data
    spark = SparkSession(sparkContext=sc)
    import_path = kwargs.get('input_data', None)
    feature_columns = kwargs.get('features', None)
    label_columns = kwargs.get('labels', None)
    id_column = kwargs.get('id', 'id')
    algorithm_params = parse_algorithm_variables(vars=kwargs.get('algo_params', None))
    standardizer = algorithm_params.pop('standardizer', False)
    data_frame = spark.read.load(
        path=import_path, format='csv', inferSchema=True,
        header=True
    ).persist()
    training_data_frame, test_data_frame = data_frame.randomSplit([0.8, 0.2])
    header_columns = training_data_frame.columns
    # Create Pipeline
    workflow = ExecuteWorkflowClassification(
        dict_params=algorithm_params, standardize=standardizer, feature_cols=feature_columns,
        label_col=label_columns, id_col=id_column
    )
    # Execute model
    model = workflow.pipeline.fit(training_data_frame)
    # Cross Validation will be included in v 1.1
    # Return result
    return model.transform(test_data_frame)