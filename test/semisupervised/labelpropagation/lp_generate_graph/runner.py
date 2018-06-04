from  pyspark import tests

from test.semisupervised.labelpropagation.lp_generate_graph import test_do_cartesian
from test.semisupervised.labelpropagation.lp_generate_graph import test__make_feature_vector
from test.semisupervised.labelpropagation.lp_generate_graph import test__compute_bfs
from test.semisupervised.labelpropagation.lp_generate_graph import test__scale_data_frame


loader = tests.unittest.TestLoader()
suite = tests.unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_do_cartesian))
suite.addTests(loader.loadTestsFromModule(test__make_feature_vector))
suite.addTests(loader.loadTestsFromModule(test__compute_bfs))
suite.addTests(loader.loadTestsFromModule(test__scale_data_frame))

runner = tests.unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)