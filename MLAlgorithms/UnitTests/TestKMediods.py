from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMediods import KMediods
from codetiming import Timer
import unittest
import os



class TestKMediods(unittest.TestCase):

    def testKMeans(self):
        data = DataRetriever("../Datasets/metadata.json")
        data.retrieveData("computerHardware")

        kValue = 15
        t = Timer()
        t.start()
        mediods = KMediods(data.getDataSet(), data.getDataClass(), data.getDescreteAttributes(),
               data.getContinuousAttributes(), data.getPredictionType(), kValue, 100)

        t.stop()
        print(f"Time: {t}")
        print(mediods)
        mediods.to_csv('out.csv', index=False)


if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()