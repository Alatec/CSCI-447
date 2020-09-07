from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.DataRetriever import DataRetriever
import pandas as pd
import unittest
import os


class TestDataRetriever(unittest.TestCase):

    ## Tests to check if the entered string returns valid JSON dataset menu
    def test_data_seperation(self):
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")

        naiveBayes = NaiveBayes()

        seperatedByClass = naiveBayes.seperateDataByClass(dataRetriever.getDataSet(), dataRetriever.getDataClass())
        print(seperatedByClass)


        self.assertEqual(dataRetriever.getDataMenu(), ["breastCancer", "glass", "iris", "soybeanSmall", "vote"]
                         , "should return list of data sets")




if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    unittest.main()