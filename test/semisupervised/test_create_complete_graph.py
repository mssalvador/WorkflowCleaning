from pyspark.sql.tests import ReusedPySparkTestCase
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from semisupervised import LP_Graph
import timeit


class TestCreate_complete_graph(ReusedPySparkTestCase):
    def setUp(self):
        helix_path = '/home/svanhmic/workspace/data/DABAI/sparkdata/csv/double_helix3.csv'
        big_data_path = '/home/svanhmic/workspace/data/DABAI/test.csv'
        self.spark_session = SparkSession(sparkContext=self.sc)
        self.helix_df = self.spark_session.read.csv(helix_path, header=True, inferSchema=True)
        self.big_data_df = (self.spark_session.read
            .csv(big_data_path, header=True, inferSchema=True)
            .drop('_c0')
            .withColumnRenamed('index','id')
            .withColumn('label', F.when(F.rand()>0.01, None).otherwise(1)))


    def test_create_complete_graph(self):

        result = LP_Graph.create_complete_graph(self.helix_df, feature_columns='x y z'.split(),
                                                id_column='id', label_column='unknown_label')
        print('Number of data points {}. Final number of points should be {}'.format(self.helix_df.count(),result.count()))
        print(result.rdd.getNumPartitions())
        timeit.timeit()

        self.assertEqual(self.helix_df.count()**2, result.count() )



    # def test_create_rdd_graph(self):
    #     result_df = LP_Graph.create_rdd_graph(
    #         self.helix_df, id_column='id', label_column='unknown_label',
    #         feature_columns='x y z'.split())
    #     # for i in result_df.take(5):
    #     #     print(i)
    #     result_df.show(5)
    #     result_df.printSchema()
    #     self.fail()