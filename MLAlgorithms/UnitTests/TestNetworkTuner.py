from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.network_tuner import network_tuner
import unittest
import os


class TestNetworkTuner(unittest.TestCase):

    def test_one_layer(self):
        layers = [*range(1, 10, 1)]


        self.assertEqual(1, 1)

    def test_two_layer(self):
        layers = [*range(1, 10, 1), *range(1, 10, 1)]

        self.assertEqual(1, 1)


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)



    unittest.main()