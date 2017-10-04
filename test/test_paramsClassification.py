from unittest import TestCase
from classification.CreateParametersClasification import ParamsClassification

class TestParamsClassification(TestCase):

    def setUp(self):

        self.result_1 = {
            'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
             'labelCol': 'label',
            'maxIter': (110, 150),
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction',
            'regParam': 0.5,
            'threshold': (0.0, 0.5),
            'tol': (1e-06, 0.01)}

        self.result_2 = {
            'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
             'labelCol': 'label',
            'maxIter': 150,
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction',
            'regParam': (0.4, 0.5),
            'threshold': (0.0, 0.5),
            'tol': (1e-06, 0.01)}

    def test_filter_single_values_to_int(self):
        p = ParamsClassification()
        case = {
            'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
            'labelCol': 'label',
            'maxIter': (110, 150),
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction',
            'regParam': 0.5,
            'threshold': (0.0, 0.5),
            'tol': (1e-06, 0.01)}
        self.assertDictEqual(p.filter_single_values_out(case),self.result_1,'It did not work')

    def test_filter_single_values_to_tuple(self):

        p = ParamsClassification()
        case = {
            'algorithm': 'LogisticRegression',
            'elasticNetParam': (0.0, 0.5),
            'fitIntercept': True,
            'labelCol': 'label',
            'maxIter': (150,150),
            'predictionCol': 'prediction',
            'probabilityCol': 'probability',
            'rawPredictionCol': 'rawPrediction',
            'regParam': (0.4, 0.5),
            'threshold': (0.0, 0.5),
            'tol': (1e-06, 0.01)}
        self.assertDictEqual(p.filter_single_values_out(case), self.result_2,'it did not work!')


