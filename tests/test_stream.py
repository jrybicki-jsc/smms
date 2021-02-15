import unittest
import pandas as pd
from graphla.streamed import naive_merge, naive_voting_merge


class StreamTests(unittest.TestCase):


    def test_naive_merge(self):
        self.assertTrue(True)

    def test_voting_merge(self):
        Y_1 = {0: [0, 19], 16384: [595, 0], 4096: [23, 1]}
        Y_2 = {1: [1, 5], 9011: [411, 20], 4096: [12, 1], 5091: [150, 33]}

        Y = naive_voting_merge(Y_1, Y_2)

        self.assertEqual(Y[0], [0, 19])
        self.assertEqual(Y[1], [1, 5])
        self.assertEqual(Y[4096], [35, 2])


