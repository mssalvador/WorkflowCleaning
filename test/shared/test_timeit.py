from unittest import TestCase
from shared.WorkflowTimer import dummy_func

class TestTimeit(TestCase):
    def test_timeit(self):
        result = dummy_func(321321)

        self.assertEqual(321321+1231, result[1])
        self.assertIsInstance(result[0],list)



