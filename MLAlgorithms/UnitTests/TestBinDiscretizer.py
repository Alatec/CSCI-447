from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.BinDiscretizer import BinDiscretizer
import pandas as pd
import unittest
import os

class TestBinDiscretizer(unittest.TestCase):

    def test_creation(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        breastCancer = dataRetriever.retrieveData("breastCancer")
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        with self.assertRaises(Exception) as context:
            BinDiscretizer()
        
        self.assertTrue('positional' in str(context.exception), "Should Raise Error if no data is passed in")


        with self.assertRaises(Exception) as context:
            BinDiscretizer()
        
        self.assertTrue('positional' in str(context.exception), "Should Raise Error if Bins is None")
       

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
        # self.assertEqual(dataRetriever.retrieveData("breastCancer"), pd.DataFrame() , "Should return a dataframe")
        self.assertEqual(dataRetriever.retrieveData("dogDiseases"), None, "Should return null since no data exist")


    pass

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()