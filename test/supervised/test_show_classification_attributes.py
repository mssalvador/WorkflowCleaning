from unittest import TestCase
from classification import ShowClassification
from pyspark.ml.tuning import CrossValidatorModel, CrossValidator

class TestShow_classification_attributes(TestCase):

    def setUp(self):
        self.cross_validator = CrossValidatorModel()


    def test_show_classification_attributes(self):
        self.assertTrue(
            ShowClassification.show_classification_attributes(CrossValidator),
            'This should be a positive result'
        )

