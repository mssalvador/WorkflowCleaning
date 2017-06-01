from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import  SparkContext

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)
class DummyData(object):
    '''
    @object this method contains dummy data for spark. The purpose of this is to test spark functions from end to end.
    '''

    def __init__(self):
        dummy_row = Row("label","value")
        self._df = sqlCtx.createDataFrame([dummy_row(i, i*100) for i in range(0, 10, 1)])

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, dic: dict, row_class: Row):
        '''
        :param dict:
        :param row_class:
        :return: pyspark dataframe with dummy data
        '''
        for key in dic.keys():
            assert key in row_class, key+str(" is not in the row")

        self._df = sqlCtx.createDataFrame([row_class(key, val) for key, val in dic.items()])

