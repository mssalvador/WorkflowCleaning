import pyspark
from pyspark.mllib.linalg.distributed import MatrixEntry
from shared.PySparkTest import PySparkTest
from semisupervised.labelpropagation.lp_preamble import preamble

DATA_PATH = '/home/michael/Data/mlaas-portal/semisupervised_distances.txt'


class TestPreamble(PySparkTest):
    def setUp(self):
        self.data = self.spark.read.csv(DATA_PATH, header=True, inferSchema=True)
        self.keyval_args = {'feature_cols': ['label_a', 'label_b', 'distance'],
                            'squared': True,
                            'datatype': 'preprocessed',
                            }

    def test_data(self):
        self.assertTrue(isinstance(self.data, pyspark.sql.DataFrame))
        self.data.show()

    def test_preamble(self):
        data, ncol = preamble(
            sc=self.spark.sparkContext,
            input_data=self.data,
            **self.keyval_args
        )
        self.assertTrue(data, pyspark.rdd)
        self.assertTrue(data.take(1)[0], MatrixEntry)
        print(data.take(5))
        # self.assertTrue()
