from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.OneHotEncoder import oneHotEncoder
import unittest
import os


class OneHotEncoder(unittest.TestCase):

    def testOneHotEncoder(self):
        dataRetriver = DataRetriever("../Datasets/metadata.json")
        glassData = dataRetriver.retrieveData("glass")

        print(oneHotEncoder(glassData.getDataSet(), glassData.getDescreteAttributes()))



if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()