from unittest import TestCase
from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification
from pyspark.sql import types as T

class TestExecuteWorkflowClassification(TestCase):

    def setUp(self):

        struct_feat = [T.StructField('f1', T.FloatType())]
        struct_lab = [T.StructField('l1', T.StringType())]



        self.workflow = ExecuteWorkflowClassification()

    def test_parameter_grid(self):
        pass

    def test_pipeline(self):
        pass

    def test_show_parameters(self):
        pass

    def test_create_custom_pipeline(self):
        pass

    def test_create_standard_pipeline(self):
        pass

    def test_list_of_equidistant_params_1(self):

        input_value = (1, 10)
        list_of_equidistants = self.workflow.list_of_equidistant_params(input_value)
        self.assertListEqual(list(list_of_equidistants), [1, 4, 7, 10])

    def test_list_of_equidistant_params_2(self):

        input_value = (1.0, 2.0)
        truth_list = [1.0, 1.333, 1.666, 2.0]
        list_of_equidistants = self.workflow.list_of_equidistant_params(input_value)
        for v, t in zip(list_of_equidistants, truth_list):
            self.assertAlmostEqual(v, t, delta=1e-3)