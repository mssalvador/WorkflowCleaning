from pyspark.ml import Estimator



# create an unsupervised classification evaluator
class ElbowEvaluation(Estimator):
    '''
        doc
    '''

    def _evaluate(self, dataset):

