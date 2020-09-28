from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
import numpy as np
import unittest
import os


class TestValueDistanceMetric(unittest.TestCase):

    def test(self):
        dataRetriver = DataRetriever("../Datasets/metadata.json")
        data = dataRetriver.retrieveData("vote")

        DistanceValueMetric(data.getDataSet(), data.getDataClass(), data.getDescreteAttributes())


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()