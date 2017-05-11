'''
Created on Feb 13, 2017

@author: svanhmic
'''
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml import Transformer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark import keyword_only



class ConvertAllToVecToMl(Transformer,HasInputCol,HasOutputCol):
    '''
        This inherrent-class converts a given vector column in a data frame to a ml-dense vector.
        Can be used in a pipeline method
    '''
    
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(ConvertAllToVecToMl, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):

        def f(s):
            return Vectors.dense([float(x) for x in s.toArray()])
        

        t = VectorUDT()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, F.udf(f, t)(in_col))
        