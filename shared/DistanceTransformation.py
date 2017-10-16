from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark import keyword_only
import numpy as np


class DistanceTransformation(Transformer, HasInputCol, HasOutputCol):
    '''

    '''

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, model=None):
        super(DistanceTransformation, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, model=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset, model):
        def computeAndInsertClusterCenter(dataset, centers):
            '''
            Insert a clusterCenter as column.
            '''

            distanceUdf = F.udf(lambda x, y: float(np.sqrt(np.sum((x - y) * (x - y)))), T.DoubleType())

            return (dataset
                    .join(F.broadcast(centers), on=(dataset["prediction"] == centers["cluster"]), how="inner")
                    .withColumn(colName="distance", col=distanceUdf(F.col("scaledFeatures"), F.col("center")))
                    .drop("cluster")
                    .drop("features")
                    .drop("v2")
                    )