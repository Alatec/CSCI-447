from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
import unittest
import os


class OneHotEncoderTest(unittest.TestCase):

    def testOneHotEncoder(self):
        dataRetriver = DataRetriever("../Datasets/metadata.json")
        glassData = dataRetriver.retrieveData("breastCancer")

        ohe = OneHotEncoder()
        encodedDataFrame = ohe.oneHotEncoder(glassData.getDataSet(), glassData.getDescreteAttributes())
        encodedDict = ohe.encodedDict

        print(encodedDataFrame)
        print(encodedDict)



if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()