'''
Created on May 15, 2017

@author: svanhmic
'''
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml import Transformer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark import keyword_only

class ComputeDistances(Transformer, HasInputCols, HasOutputCol):

    '''
        This inherrent-class converts a given vector column in a data frame to a ml-dense vector.
        Can be used in a pipeline method
    '''

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(ComputeDistances, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):

        t = VectorUDT()
        out_col = self.getOutputCol()
        print(self.getInputCols())
        if isinstance(self.getInputCols(), list):
            print("is list")
        else:
            print("is not list")
        #in_col = dataset[self.getInputCol()]
        #return dataset.withColumn(out_col, F.udf(f, t)(in_col))