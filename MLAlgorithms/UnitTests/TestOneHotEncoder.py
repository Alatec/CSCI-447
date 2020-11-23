from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
import unittest
import os


class OneHotEncoderTest(unittest.TestCase):

    def testOneHotEncoder(self):
        dataRetriver = DataRetriever("../Datasets/metadata.json")
        glassData = dataRetriver.retrieveData("breastCancer")
        data = glassData.getDataSet()
        unknown = glassData.getDataClass()
        train = data.sample(n=6, random_state=69)
        test = data.sample(n=6, random_state=420)

        ohe = OneHotEncoder()
        encodedDataFrame = ohe.train_fit(train, glassData.getDescreteAttributes())
        encodedDict = ohe.encodedDict

        encodedTest = ohe.fit(test)

        # print(encodedDataFrame)
        # print(encodedDict)
        print("=============Train============")
        print(encodedDataFrame[unknown])
        print(train[unknown])
        print("=============Test=============")
        print(encodedTest[unknown])
        print(test[unknown])



if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()