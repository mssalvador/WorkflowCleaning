from unittest import TestCase
from unittest.mock import patch
from pandas.util.testing import assert_frame_equal
from pyspark.sql import SparkSession
from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow
TEST_DATA_PATH = '/home/svanhmic/workflow/data/DABAI/sparkdata/csv/test.csv'


class TestExecuteWorkflow(PySparkTestCase):

    def setUp(self):
        spark_session = SparkSession(sparkContext=self.sc)
        workflow = ExecuteWorkflow()

    #def test_params(self):
    #    self.fail()

    #def test_params(self):
    #    self.fail()

    #def test_gen_cluster_center(self):
    #    self.fail()

    def test_vector_scale(self):
        pass

