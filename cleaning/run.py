import pyspark
import pyspark.sql.types as T
import pyspark.sql.functions as F
# from shared.WorkflowLogger import logger_info_decorator


# @logger_info_decorator
def run(sc: pyspark.SparkContext, **kwargs):

    from shared.parse_algorithm_variables import parse_algorithm_variables

    from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow
    from cleaning.ShowCleaning import ShowResults
    # Initialization phase v.1.0
    import_path = kwargs.get('input_data', None)
    feature_columns = kwargs.get('features', None)
    label_columns = kwargs.get('labels', 'k')
    id_column = kwargs.get('id', 'id')
    algorithm_params = parse_algorithm_variables(kwargs.get('algo_params', None))
    standardizer = algorithm_params.get('standardizer', False)

    spark_session = pyspark.sql.SparkSession(sc)
    label_schema = [T.StructField(l, T.StringType(), False) for l in label_columns]
    id_schema = [T.StructField(idx, T.StringType(), False) for idx in id_column]
    feature_schema = [T.StructField(f, T.DoubleType(), False) for f in feature_columns]
    training_data_schema = T.StructType(id_schema+label_schema+feature_schema)
    training_data_frame = spark_session.read.load(
        path=import_path, format='csv', schema=training_data_schema)
    training_data_frame = training_data_frame.na.drop()
    # training_data_frame.show()
    cleaning_workflow = ExecuteWorkflow(
        dict_params=algorithm_params, cols_features=feature_columns,
        cols_labels=label_columns, standardize=standardizer
    )

    training_model = cleaning_workflow.execute_pipeline(training_data_frame)
    clustered_data_frame = cleaning_workflow.apply_model(
        sc=sc, model=training_model, data_frame=training_data_frame)

    # print(algorithm_params)
    show_result = ShowResults(
        sc=sc, dict_parameters=algorithm_params,
        list_features=feature_columns, list_labels=label_columns)

    all_info_df = show_result.prepare_table_data(clustered_data_frame, **algorithm_params)
    d_point = 'data_points'
    new_struct = F.struct([id_column[0], *feature_columns, 'distance', 'is_outlier']).alias(d_point)

    buket_df = show_result.create_buckets(sc, all_info_df, **algorithm_params)

    return (all_info_df
        .select(F.col(algorithm_params['predictionCol']), new_struct)
        .groupBy(F.col(algorithm_params['predictionCol'])).agg(
        F.count(algorithm_params['predictionCol']).alias('amount'),
        F.sum(F.col(d_point+".is_outlier")).alias('percentage_outlier'),
        F.collect_list(d_point).alias(d_point))
        .join(other=buket_df, on=algorithm_params['predictionCol'], how='inner')
        .withColumn('percentage_outlier', 100 * F.col('percentage_outlier') / F.col('amount'))
    )




