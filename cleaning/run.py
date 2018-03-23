import pyspark
import pyspark.sql.types as T
# from shared.WorkflowLogger import logger_info_decorator

# @logger_info_decorator
def run(sc: pyspark.SparkContext, **kwargs):

    from shared.parse_algorithm_variables import parse_algorithm_variables
    from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow
    from cleaning.ShowCleaning import ShowResults

    # Initialization phase v.1.0
    import_path = kwargs.get('input_data', None)
    feature_columns = kwargs.get('features', None)
    label_columns = kwargs.get('labels', None)
    id_column = kwargs.get('id', 'id')
    algorithm_params = parse_algorithm_variables(
        vars=kwargs.get('algo_params', None)
    )
    standardizer = algorithm_params.get('standardizer', False)
    spark_session = pyspark.sql.SparkSession(sc)

    # label_schema = create_sub_schema(label_columns, type='label')
    # id_schema = create_sub_schema(id_column, type='id')
    # feature_schema = create_sub_schema(feature_columns, type='feature')
    # all_structs = list(filter(lambda x: x != None, id_schema+label_schema+feature_schema))
    # training_data_schema = T.StructType(all_structs)

    training_data_frame = spark_session.read.load(
        path=import_path, format='csv', inferSchema=True,
        header=True
    ).persist()
    training_data_frame.take(1)
    #training_data_frame.show()
    cleaning_workflow = ExecuteWorkflow(
        dict_params=algorithm_params, cols_features=feature_columns,
        cols_labels=label_columns, standardize=standardizer
    )
    training_model = cleaning_workflow.execute_pipeline(
        data_frame=training_data_frame
    )
    clustered_data_frame = cleaning_workflow.apply_model(
        sc=sc, model=training_model, data_frame=training_data_frame
    )
    # clustered_data_frame.show()
    show_result = ShowResults(
        id=id_column[0], list_features=feature_columns,
        list_labels=label_columns, **algorithm_params
    )
    all_info_df = show_result.prepare_table_data(
        dataframe=clustered_data_frame, **algorithm_params
    )
    # all_info_df.show()
    d_point = 'data_points'

    output_df = show_result.arrange_output(
        sc=sc, dataframe=all_info_df,
        data_point_name=d_point, **algorithm_params
    )
    training_data_frame.unpersist()
    return output_df


def create_sub_schema(columns, type='label'):
    types = {'label': T.StringType(),
             'id' : T.IntegerType(),
             'feature': T.DoubleType()
             }
    if isinstance(columns, list):
        return [T.StructField(
            name=column, dataType=types[type],
            nullable=False) for column in columns
        ]
    elif isinstance(columns, str):
        return [T.StructField(
            name=columns, dataType=types[type],
            nullable=False)
        ]
    else:
        return [None]
