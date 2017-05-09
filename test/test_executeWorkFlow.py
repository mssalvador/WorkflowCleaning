from unittest import TestCase
from sample.ExecuteWorkflow import ExecuteWorkFlow
import pyspark.ml.clustering as clusters

class TestExecuteWorkFlow(TestCase):

    def setUp(self):
        self.workflow = ExecuteWorkFlow()

    def test_set_empty_model(self): #model has not been set
        self.assertRaises(AssertionError,self.workflow.model)

    def test_set_kmeans_model(self): #model has been set to Kmeans
        self.workflow.model="KMeans"
        self.assertIn(self.workflow.model,clusters.__all__)

    def test_set_wrong_model(self): #model has been set to LogisticRegression!!!CANNOT HAPPEN!!!
        self.assertRaises(AssertionError, setattr, self.workflow, "model", "LogsticRegression")


