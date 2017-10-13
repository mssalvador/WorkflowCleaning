from pyspark.ml.tuning import ValidatorParams
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import  functions as F
from pyspark.ml import Estimator

from pyspark import keyword_only


# create an unsupervised classification evaluator
class ElbowEvaluation(Estimator, ValidatorParams):
    '''
        doc
    '''

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                 seed=None):
        super(ElbowEvaluation, self).__init__()
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    computeDistanceToCenterUdf = F.udf(lambda x, y: (x - y) * (x - y), VectorUDT())

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)

        for j in range(numModels):
            model = est.fit(dataset, epm[j])
            model.

            metric = eva.evaluate(model.transform(dataset, epm[j]))
            metrics[j] += metric
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics))

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies creates a deep copy of
        the embedded paramMap, and copies the embedded and extra parameters over.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        newTVS = Params.copy(self, extra)
        if self.isSet(self.estimator):
            newTVS.setEstimator(self.getEstimator().copy(extra))
        # estimatorParamMaps remain the same
        if self.isSet(self.evaluator):
            newTVS.setEvaluator(self.getEvaluator().copy(extra))
        return newTVS