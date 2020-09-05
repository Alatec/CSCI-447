from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.BinDiscretizer import BinDiscretizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

        with self.assertRaises(TypeError):
            BinDiscretizer(breastCancer["clumpThickness"], bins=None)
     
    ## Tests to check if the entered string returns valid JSON dataset menu
    def test_calc_bins(self):
        #Initialization
        dataRetriever = DataRetriever("../Datasets/metadata.json")
        breastCancer = dataRetriever.retrieveData("breastCancer")
        continousAttributes = ["clumpThickness", "uniformityOfCellSize", "uniformityOfCellShape",
                      "marginalAdhesion", "singleEpithelialCellSize", "bareNuclei", "blandChromatin", "normalNucleoli", "mitoses", "class"]

        bd = BinDiscretizer(breastCancer["clumpThickness"], bins=8)
        fitted_data = bd.train_fit()
        
        numpy_bins = np.histogram_bin_edges(breastCancer["clumpThickness"], bins=8)
        
        self.assertEqual(np.allclose(bd.bin_edges, numpy_bins[1:]),True, "Should produce the same bins as np.histogram")
        
        #Numpy digitize has this weird shenanigan where values of array.max() are considered the next highest bin:
        #https://stackoverflow.com/questions/4355132/numpy-digitize-returns-values-out-of-range
        #These lines convert the array returned to consider the max values as part of the right most bin
        numpy_digitize = np.digitize(breastCancer["clumpThickness"], numpy_bins)
        max_vals = np.asarray(np.where(numpy_digitize==numpy_digitize.max())).flatten()
        numpy_digitize[max_vals] = numpy_digitize.max() - 1
        print("Shape", breastCancer.shape)
        self.assertEqual(np.allclose(numpy_digitize, fitted_data), True, "Should produce the same results as np.digitize")




    pass

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()