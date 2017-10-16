'''
Created on Oct 13, 2017

@author: svanhmic
'''
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml import Transformer
from pyspark.sql import functions as F
from pyspark import keyword_only
from pyspark.ml.param import Params, Param, TypeConverters


class CastInPipeline(Transformer, HasInputCol):
    '''
        This inherrent-class converts a given vector column in a data frame to a ml-dense vector.
        Can be used in a pipeline method
    '''

    applicable_casts = ['intstring',
                        'intfloat',
                        'intdouble',
                        'doublefloat',
                        'floatdouble',
                        'stringdouble',
                        'stringint']

    castTo = Param(
        Params._dummy(),
        'castTo',
        'Indicates the what we want to cast to.',
        typeConverter=TypeConverters.toString
    )

    @keyword_only
    def __init__(self, inputCol=None, castTo=None,):

        if castTo not in ['string', 'int', 'float', 'double', 'boolean']:
            raise TypeError('new type must be a valid type!')

        super(CastInPipeline, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, castTo=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setCastTo(self, value):
        """
        Sets the casted value to
        :return:
        """
        if value in ['string', 'int', 'float', 'double', 'boolean']:
            return self._set(castTo=value)
        else:
            raise TypeError('new type must be a valid type!')

    def getCastTo(self):
        return self.getOrDefault(self.castTo)


    def _transform(self, dataset):

        column_types = dict(dataset.dtypes)
        if str(column_types[self.getInputCol()])+str(self.getCastTo()) not in self.applicable_casts:
            raise Exception(
                'The desired conversion from {} to {}, cannot be applied, sorry!'
                    .format(column_types[self.getInputCol()], self.getCastTo())
            )
        return dataset.withColumn(
            self.getInputCol(),
            F.col(self.getInputCol()).cast(self.getCastTo()))