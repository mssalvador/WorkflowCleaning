from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkContext
from random import random, randint
from pyspark.sql.types import FloatType, StructType, StructField, StringType


class DummyData(object):
    '''
    @object this method contains dummy data for spark. The purpose of this is to test spark functions from end to end.
    '''

    def __init__(self, sc: SparkContext, number_of_samples=50):

        self.sc = sc
        self.sqlCtx = SQLContext.getOrCreate(sc)
        dummy_row = Row("label", "x", "y", "z")
        list_of_struct = [StructField(dummy_row[0], StringType())]+[StructField(i, FloatType()) for i in dummy_row[1:]]
        schema = StructType(list_of_struct)
        self._df = self.sqlCtx.createDataFrame([dummy_row(randint(0, 5), 3*random(), 4*random(), 5*random()) for _ in range(0, number_of_samples, 1)],schema)

    def __del__(self):
        print("destroyed")
        self.sc.stop()



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

        self._df = self.sqlCtx.createDataFrame([row_class(key, val) for key, val in dic.items()])

    def create_outliers(self, column, outlier_factor):

        def make_possible_outlier(x):
            if random() > 0.5:
                return x*outlier_factor
            else:
                return x

        is_outlier = F.udf(lambda x: make_possible_outlier(x), FloatType())

        self._df = self._df.withColumn(column, is_outlier(column))

    def show(self):
        self._df.show()
