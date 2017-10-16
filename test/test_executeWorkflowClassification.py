from unittest import TestCase
from pyspark import SparkContext, SQLContext
from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from shared.ConvertAllToVecToMl import ConvertAllToVecToMl

class TestExecuteWorkflowClassification(TestCase):

    def setUp(self):

        sc = SparkContext.getOrCreate()
        sql_context = SQLContext(sc)
        struct_feat = [T.StructField('f1', T.FloatType())]
        struct_lab = [T.StructField('l1', T.StringType())]

        self.default_param_dict = {
            'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
            'labelCol': 'label',
            'maxIter': (100, 150),
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction',
            'regParam': (0.0, 0.5),
            'threshold': (0.0, 0.5),
            'tol': (1e-06, 0.01)
        }

        self.default_features = [
            T.StructField('AarsVaerk_1', T.DoubleType(), True),
            T.StructField('AarsVaerk_2', T.DoubleType(), True),
            T.StructField('AarsVaerk_3', T.DoubleType(), True)
        ]
        self.default_standard = True

        self.workflow = ExecuteWorkflowClassification(
            self.default_param_dict,
            self.default_standard,
            self.default_features
        )

    def test_parameter_grid(self):
        pass

    def test_pipeline(self):
        pass

    def test_show_parameters(self):
        pass

    def test_create_custom_pipeline(self):
        pass

    def test_create_standard_pipeline(self):
        tested_pipeline = self.workflow.pipeline
        self.assertTrue(
            isinstance(
                tested_pipeline,
                Pipeline
            ),
            'This is not a pipeline!'
        ) # Test if we actually get a pipeline

        # should also have a test that shows each stage is correct
        stages = [VectorAssembler(),
                  ConvertAllToVecToMl(),
                  StandardScaler(),
                  LogisticRegression()]


    def test_list_of_equidistant_params_1(self):

        input_value = {'dummy_tuple': (1, 10)}
        list_of_equidistants = self.workflow.generate_equidistant_params(input_value)
        list_of_ranges = list(list_of_equidistants)[0][1]
        self.assertListEqual(
            list_of_ranges.tolist(),
            [1, 5, 10]
        )
