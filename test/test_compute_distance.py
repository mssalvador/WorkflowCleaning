from unittest import TestCase
import shared.ComputeDistances as compute_dist
import numpy as np
import math


class TestCompute_distance(TestCase):
    def setUp(self):
        self.distance = None
        self.dummy_point = np.array([1, 1, 2])
        self.dummy_center = np.array([0, 0, 0])

    def test_compute_distance(self):


        self.distance = compute_dist.compute_distance(self.dummy_point, self.dummy_center)
        self.assertIsInstance(self.distance,float,str(type(self.distance)))
        self.assertEqual(self.distance, math.sqrt(6),str(self.distance)+" is not equal to: "+str(math.sqrt(3)))

