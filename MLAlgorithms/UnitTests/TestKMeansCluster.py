from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMeansClustering import KMeans
from codetiming import Timer
import unittest
import os



class TestKMeansCluster(unittest.TestCase):

    def testKMeans(self):
        data = DataRetriever("../Datasets/metadata.json")
        data.retrieveData("computerHardware")

        kValue = 15
        t = Timer()
        t.start()
        centroids = KMeans(data.getDataSet(), data.getDataClass(), data.getDescreteAttributes(),
               data.getContinuousAttributes(), data.getPredictionType(), kValue, 100)

        t.stop()
        print(f"Time: {t}")
        print(centroids)
        centroids.to_csv('KMeans.csv', index=False)


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()