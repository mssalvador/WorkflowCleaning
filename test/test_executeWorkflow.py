from unittest import TestCase
from sample.ExecuteWorkflow import ExecuteWorkflow
from pyspark.ml.pipeline import Pipeline

TEST_DICT = {'features': ('AarsVaerk_1', 'AarsVaerk_2', 'AarsVaerk_3', 'AarsVaerk_4', 'AarsVaerk_5', 'AarsVaerk_6', 'AarsVaerk_7', 'AarsVaerk_8', 'AarsVaerk_9', 'AarsVaerk_10', 'AarsVaerk_11', 'AarsVaerk_12', 'AarsVaerk_13', 'AarsVaerk_14', 'AarsVaerk_15', 'medArb_1', 'medArb_2', 'medArb_3', 'medArb_4', 'medArb_5', 'medArb_6', 'medArb_7', 'medArb_8', 'medArb_9', 'medArb_10', 'medArb_11', 'medArb_12', 'medArb_13', 'medArb_14', 'medArb_15', 'avgVarighed', 'totalAabneEnheder', 'totalLukketEnheder', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5', 'rank_6', 'rank_7', 'reklamebeskyttet'),
             'initialstep': 43,
             'standardize': True,
             'clusters': 24,
             'model': 'KMeans',
             'initialmode': 'random',
             'prediction': 'predict',
             'iterations': 27
             }


class TestExecuteWorkflow(TestCase):

    def setUp(self):
        self.workflow = ExecuteWorkflow()

    def test_params(self):
        parameters = {"model":"KMeans", "type":"random", "standardize":True}
        self.workflow.params = parameters
        self.assertDictEqual(self.workflow.params, parameters)


    def test_construct_pipeline(self):
        self.fail()

    #def test_run(self):
    #    self.fail()
