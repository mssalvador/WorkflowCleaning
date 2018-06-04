import pyspark
from pyspark import tests
from pyspark import sql
from pyspark.ml import  linalg as ml_linalg
from labelpropagation.lp_data_gen import generate_springs
from labelpropagation.lp_generate_graph import do_cartesian
import numpy as np


class TestDo_cartesian(tests.ReusedPySparkTestCase):
    def setUp(self):
        z = np.linspace(0, 3, 10)
        self.test_data = generate_springs(2.5, 1, z, -z)

    def test_do_cartesian(self):
        spark_session = sql.SparkSession(self.sc)
        string_rdd = self.sc.parallelize(self.test_data).map(
            lambda x: pyspark.Row(id=x[0], label=x[1], vector=ml_linalg.DenseVector(x[2])))
        string_df = string_rdd.toDF()
        test_demon = do_cartesian(sc=self.sc, df=string_df, id_col='id', feature_col='vector')
        check_diagonal = test_demon.filter(lambda x: x.i == x.j).map(lambda x: x.value).collect()
        for diag in check_diagonal:
            self.assertEqual(1.0, diag)
