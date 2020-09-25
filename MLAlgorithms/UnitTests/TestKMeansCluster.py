from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Project2.KMeansClustering import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os


class TestKMeansCluster(unittest.TestCase):

    def testKMeans(self):
        dataRetriver = DataRetriever("../Datasets/metadata.json")
        glassData = dataRetriver.retrieveData("vote")

        kValue = 3
        print(KMeans(glassData.getDataSet(), kValue))


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()