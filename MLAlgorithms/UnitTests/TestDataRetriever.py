from MLAlgorithms.Utils.DataRetriever import DataRetriever
import pandas as pd
import unittest

class TestDataRetriever(unittest.TestCase):

    def test_creation(self):
        dataRetriever = DataRetriever("bad path")
        self.assertEqual(dataRetriever, None, "Should return None when a bad path is entered")

    ## Tests to check if the entered string returns valid JSON dataset menu
    def test_menu(self):
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        self.assertEqual(dataRetriever.getDataMenu(), ["breastCancer", "glass", "iris", "soybeanSmall", "vote"]
                         , "should return list of data sets")

    ## Tests to check if a given object is there
    def test_existence(self):
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        self.assertEqual(dataRetriever.hasData("breastCancer"), True, "Should have breast cancer data")
        self.assertEqual(dataRetriever.hasData("dogDiseases"), False, "Shouldn't have dog disease data")

    ## Tests to ensure that we can grab an item from the JSON menu
    def test_data_retrieval(self):
        dataRetriever = DataRetriever("../Datasets/metadata.json")

        # This test is failing because the test itself isn't working
        self.assertEqual(dataRetriever.retrieveData("breastCancer"), pd.DataFrame() , "Should return a dataframe")
        self.assertEqual(dataRetriever.retrieveData("dogDiseases"), None, "Should return null since no data exist")


    pass

if __name__ == '__main__':
    unittest.main()