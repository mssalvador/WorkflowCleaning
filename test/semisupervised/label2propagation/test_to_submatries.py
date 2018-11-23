from pyspark.sql import SparkSession
from pyspark.sql import dataframe
from pyspark import tests
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from semisupervised.label2propagation.split_to_submatrix import to_submatries
import numpy as np

class TestTo_submatries(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.spark = SparkSession(self.sc)

        self.val = 10
        sample = [np.nan, 0, 1]
        prob = [.8, .1, .1]
        data = [{"id":i,
                 "feature": Vectors.dense(np.random.rand(self.val)),
                 "label": float(np.random.choice(sample, 1, replace=True, p=prob))} for i in range(self.val)]
        schema = T.StructType([
            T.StructField("id", T.IntegerType()),
            T.StructField("feature", VectorUDT()),
            T.StructField("label", T.FloatType())])

        self.input = self.spark.createDataFrame(data,schema)

    def test_to_submatries(self):
        broadcast_l = self.sc.broadcast(self.input.filter(~F.isnan("label")).count())
        # self.input.show()
        key_val = {"feature": "feature", "label": "label", "id": "id"}
        T_ll, T_lu, T_ul, T_uu = to_submatries(self.input, broadcast_l=broadcast_l, **key_val)
        # T_ul.show()
        T_uu.show(truncate=False)
        T_ll.show()

        # Is our output dataframes?
        self.assertIsInstance(T_ul, dataframe.DataFrame)
        self.assertIsInstance(T_uu, dataframe.DataFrame)

        # Has our left matrix dvs. T_ul output the correct length?
        self.assertIs(broadcast_l.value, len(T_ul.select("left").take(1)[0]["left"]))

        # Has our rigth matrix dvs. T_ul output the correct length?
        self.assertIs(self.val-broadcast_l.value, len(T_uu.select("right").take(1)[0]["right"]))

        # Does a random element in the original correspond to the new values?
        samples = list(map(lambda x: x["id"], T_uu.select('id').collect()))
        r = np.random.choice(samples, 1)
        self.assertListEqual(self.input.filter(F.col("id") == int(r)).collect()[0]["feature"][:broadcast_l.value].tolist(),
                             T_ul.filter(F.col("id") == int(r)).collect()[0]["left"].tolist()
                             )

        self.assertListEqual(self.input.filter(F.col("id") == int(r)).collect()[0]["feature"][broadcast_l.value:].tolist(),
                             T_uu.filter(F.col("id") == int(r)).collect()[0]["right"].tolist()
                             )
