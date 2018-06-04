from pyspark import tests
import numpy as np
from labelpropagation import lp_data_gen


class TestGenerate_springs(tests.ReusedPySparkTestCase):
    def setUp(self):
        self.a = 2.5
        self.n = 10
        start = 0
        stop = 3
        l_space = np.linspace(start=start, stop=stop, num=self.n)
        self.linspaces = [l_space, -l_space]

    def test_generate_springs(self):
        # print(self.linspaces)
        computed_data = lp_data_gen.generate_springs(self.a, 1, *self.linspaces)
        # for p in computed_data:
        #     print(p)
        self.assertEqual(len(computed_data), self.n*2)

    def test_generate_string_data(self):
        spring = lp_data_gen.generate_string_data(self.linspaces[0], self.a, 0)
        self.assertEqual(len(spring), self.n)

