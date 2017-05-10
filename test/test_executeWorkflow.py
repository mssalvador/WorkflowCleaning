from unittest import TestCase
from sample.ExecuteWorkflow import ExecuteWorkflow

class TestExecuteWorkflow(TestCase):

    def setUp(self):
        self.workflow = ExecuteWorkflow()

    def test_params(self):
        parameters = {"model":"KMeans", "type":"random", "standardize":True}
        self.workflow.params = parameters
        self.assertDictEqual(self.workflow.params, parameters)

    def test_params(self):
        self.parameters = ["KMeans","random",True]
        self.a


    #def test_construct_pipeline(self):
     #   self.fail()

    #def test_run(self):
    #    self.fail()
