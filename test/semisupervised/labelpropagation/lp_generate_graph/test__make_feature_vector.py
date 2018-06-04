from pyspark import tests
from pyspark.sql import SparkSession
from semisupervised.labelpropagation import lp_generate_graph
import numpy as np
import pyspark



class Test_make_feature_vector(tests.ReusedPySparkTestCase):
    def setUp(self):
        spark = SparkSession(sparkContext=self.sc)
        data = np.random.randint(0,9,[100,10])
        row = pyspark.Row('id', *['feature_'+str(i) for i in range(9)])
        self.x_rdd = self.sc.parallelize(list(map(lambda x: row(*x), data.tolist())))
        self.x_df = spark.createDataFrame(self.x_rdd)

    def test__make_feature_vector(self):
        self.fail()

