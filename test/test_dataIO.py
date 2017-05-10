from unittest import TestCase
from sample.DataIO import DataIO
from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext("local[*]","test cleaning workflow")
sqlContext = SQLContext(sc)

class TestDataIO(TestCase):

    def setUp(self):
        data_io = DataIO()

    def test_import_features_df(self):
        self.fail()

    def test_import_companies_df(self):
        self.fail()

    def test_show_features(self):
        self.fail()

    def test_get_latest_company(self):
        self.fail()

    def test_mergeCompanyFeatureData(self):
        self.fail()


