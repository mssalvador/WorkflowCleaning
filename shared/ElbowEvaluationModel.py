
from pyspark.ml.tuning import ValidatorParams


class ElbowEvaluationModel(Model, ValidatorParams):
    """
    .. note:: Experimental

    Model from train validation split.

    .. versionadded:: 2.0.0
    """

    def __init__(self, bestModel, validationMetrics=[]):
        super(TrainValidationSplitModel, self).__init__()
        #: best model from cross validation
        self.bestModel = bestModel
        #: evaluated validation metrics
        self.validationMetrics = validationMetrics

    def _transform(self, dataset):
        return self.bestModel.transform(dataset)

    def copy(self, extra=None):
        """
        Creates a copy of this instance with a randomly generated uid
        and some extra params. This copies the underlying bestModel,
        creates a deep copy of the embedded paramMap, and
        copies the embedded and extra parameters over.
        And, this creates a shallow copy of the validationMetrics.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        bestModel = self.bestModel.copy(extra)
        validationMetrics = list(self.validationMetrics)
        return TrainValidationSplitModel(bestModel, validationMetrics)

