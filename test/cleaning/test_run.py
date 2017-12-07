from pyspark.tests import ReusedPySparkTestCase, PySparkTestCase
from cleaning.run import run
from pyspark.ml import clustering




class TestRun(PySparkTestCase):

    def test_run(self):
        print(clustering.__all__)
        self.fail()

    def test_parse_algorithm_variables(self):
        self.fail()