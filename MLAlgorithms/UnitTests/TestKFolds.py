from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os 

class TestBinDiscretizer(unittest.TestCase):

    def test_train_test_sizes(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")
        breastCancer = dataRetriever.getDataSet()
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        for test, train in KFolds(breastCancer, 10):
           self.assertEqual(len(test)+len(train), len(breastCancer))
    
    def test_train_test_independence(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")
        breastCancer = dataRetriever.getDataSet()
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        for test, train in KFolds(breastCancer, 10):
            #https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
            self.assertFalse(bool(set(test.index)&set(train.index)))

    def test_test_set_coverage(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")
        breastCancer = dataRetriever.getDataSet()
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        tested_vals = []

        #Add a dummy index to the dataset so we can see which rows are selected each test
        breastCancer["dummyIndex"] = np.arange(len(breastCancer)) + 1
        
        for test, train in KFolds(breastCancer, 10):
           tested_vals.extend(test["dummyIndex"])
           
        
        self.assertTrue(set(tested_vals)==set(breastCancer["dummyIndex"]))

    def test_proper_number_of_folds(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")
        breastCancer = dataRetriever.getDataSet()
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        iterations = 0
        for test, train in KFolds(breastCancer, 10):
            iterations+=1
        
        self.assertEqual(iterations,10)


    def test_stratisfied(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        dataRetriever.retrieveData("breastCancer")
        breastCancer = dataRetriever.getDataSet()
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        iterations = 0
        for test, train in KFolds(breastCancer, 10, stratisfied=True):
            print("TestLen 2",len(test[test['class']==2]), "TestLen 4",len(test[test['class']==4]))
            print("TrainLen 2",len(train[train['class']==2]), "TrainLen 4",len(train[train['class']==4]))
            iterations+=1

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()