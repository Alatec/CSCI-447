from MLAlgorithms.Utils.HammingDistance import calculateHammingDistance
import numpy as np
import unittest
import os


class HammingDistance(unittest.TestCase):

    def testHammingDistanceInt(self):
        vect1 = np.asarray([0, 0, 1, 0, 1, 1])
        vect2 = np.asarray([1, 0, 1, 1, 1, 1])

        self.assertEqual(calculateHammingDistance(vect1, vect2), 2, "The hamming distance should be 2")

    def testHammingDistanceBool(self):
        vect1 = np.asarray([False, False, True, False, True, True])
        vect2 = np.asarray([True, False, True, True, True, True])

        self.assertEqual(calculateHammingDistance(vect1, vect2), 2, "The hamming distance should be 2")

    def testHammingDistanceDifferentSizes(self):
        vect1 = np.asarray([False, False, True, False])
        vect2 = np.asarray([True, False, True, True, True, True])

        self.assertEqual(calculateHammingDistance(vect1, vect2), -1, "Trying to calculate the hamming distance between vectors of different length should return -1")


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()