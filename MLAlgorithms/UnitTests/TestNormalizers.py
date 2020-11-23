from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer

from MLAlgorithms.Errors.UtilError import TrainTestColumnMismatch, UntrainedUtilityError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os 

class TestNormalizers(unittest.TestCase):

    def test_col_mismatch(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        breastCancer = dataRetriever.retrieveData("breastCancer")
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        rn = RangeNormalizer(breastCancer[continousAttributes])
        sn = StandardNormalizer(breastCancer[continousAttributes])

        rn.train()
        sn.train()

        #Test range normalizer
        with self.assertRaises(TrainTestColumnMismatch):
            rn.fit(breastCancer["mitoses"])
        
        #Test standard normalizer
        with self.assertRaises(TrainTestColumnMismatch):
            sn.fit(breastCancer["mitoses"])

    def test_untrained(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        breastCancer = dataRetriever.retrieveData("breastCancer")
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        rn = RangeNormalizer(breastCancer[continousAttributes])
        sn = StandardNormalizer(breastCancer[continousAttributes])

        
        #Test range normalizer
        with self.assertRaises(UntrainedUtilityError):
            rn.fit(breastCancer[continousAttributes])
        
        #Test standard normalizer
        with self.assertRaises(UntrainedUtilityError):
            sn.fit(breastCancer[continousAttributes])
        
    def test_range_normalizer_bounds(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        breastCancer = dataRetriever.retrieveData("breastCancer")
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        rn = RangeNormalizer(breastCancer[continousAttributes])
        
        fitted = rn.train_fit()

        #Check if the mins/maxes of all the fitted columns are 0/1, respectively
        self.assertEqual(np.allclose(np.ones(fitted.shape[1]), fitted.max()), True)
        self.assertEqual(np.allclose(np.zeros(fitted.shape[1]), fitted.min()), True)




    pass

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()