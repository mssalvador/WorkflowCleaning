from  pyspark import tests

from test.semisupervised.labelpropagation.lp_generate_graph import test_do_cartesian

loader = tests.unittest.TestLoader()
suite = tests.unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_do_cartesian))

runner = tests.unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)