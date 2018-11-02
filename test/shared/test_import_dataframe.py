import os
from pyspark.sql import DataFrame

from shared.PySparkTest import PySparkTest
from shared.data_import.import_data import import_dataframe


class TestImport_dataframe(PySparkTest):
    def setUp(self):
        self.path = '/home/michael/Data/mlaas-portal/double_helix3.csv'
        self.err_path = self.path.replace('3', '2')

    def test_import_dataframe(self):
        self.assertTrue(os.path.exists(self.path), 'This is not true {}'.format(self.path))
        self.assertFalse(os.path.exists(self.err_path), 'This is path: {}, should not be there'.format(self.err_path))

    def test_load_data_frame(self):
        df = import_dataframe(spark_context=self.spark, data=self.path)
        self.assertIsInstance(df, DataFrame, 'df is not a {}, but {}'.format(type(df), type(DataFrame)))
