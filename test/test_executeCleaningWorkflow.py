from unittest import TestCase
from unittest.mock import patch
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow

class TestExecuteWorkflow(TestCase):
    #def test_params(self):
    #    self.fail()

    #def test_params(self):
    #    self.fail()

    def test_gen_gaussians_center(self):

        k = 5
        gaussian = pd.DataFrame({'mean': [np.random.rand(k) for _ in range(k)], 'cov': [np.random.rand(k, k) for _ in range(k)]})
        foo = pd.DataFrame({'prediction': range(k)})

        combined = ExecuteWorkflow.gen_gaussians_center(k, gaussians=gaussian)
        print(assert_frame_equal(combined, pd.concat([gaussian, foo], axis=1)))

    #def test_gen_cluster_center(self):
    #    self.fail()
