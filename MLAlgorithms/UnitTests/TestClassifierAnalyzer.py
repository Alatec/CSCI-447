# from MLAlgorithms.Utils.DataRetriever import DataRetriever
# from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
# from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer

# from MLAlgorithms.Errors.UtilError import TrainTestColumnMismatch, UntrainedUtilityError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os 

class TestClassifierAnalyzer(unittest.TestCase):

    def test_create_confusion_matrix(self):
        #Initialization
        truth = np.random.choice([0,1,2], 100,replace=True)
        pred = np.random.choice([0,1,2], 100,replace=True)

        ca = ClassifierAnalyzer(truth, pred)
        
        self.assertEqual(len(ca.confusion_matrix.shape),2)

    def test_single_class_methods(self):
        #Initialization
        truth = np.array([1,1,0,0,1,1,1,1,1,0])
        pred  = np.array([0,1,1,0,0,1,1,1,0,0])

        ca = ClassifierAnalyzer(truth, pred)
        
        #Check Accuracy
        self.assertAlmostEqual(ca.calc_accuracy(),0.6, places=4)

        #check precision
        self.assertAlmostEqual(ca.calc_precision(),0.8, places=4)

        #check recall
        self.assertAlmostEqual(ca.calc_recall(), 4/7, places=4)

        #check f1
        self.assertAlmostEqual(ca.calc_f1_score(), 2/3, places=4)

    def test_micro_averages(self):
        #Initialization
        truth = np.array([1,2,3,1,2,3,1,2,3,1])
        pred  = np.array([2,2,2,3,1,3,2,3,1,1])

        ca = ClassifierAnalyzer(truth, pred)

        #Check Accuracy
        self.assertAlmostEqual(ca.calc_accuracy("micro"),0.3, places=4)

        #check precision
        self.assertAlmostEqual(ca.calc_precision("micro"),0.3, places=4)

        #check recall
        self.assertAlmostEqual(ca.calc_recall("micro"), 0.3, places=4)

        #check f1
        self.assertAlmostEqual(ca.calc_f1_score("micro"), 0.3, places=4)

    def test_macro_averages(self):
        #Initialization
        truth = np.array([1,2,3,1,2,3,1,2,3,1])
        pred  = np.array([2,2,2,3,1,3,2,3,1,1])

        ca = ClassifierAnalyzer(truth, pred)
        

        #Check Accuracy
        self.assertAlmostEqual(ca.calc_accuracy("macro"),0.5333333333333333, places=4)

        #check precision
        self.assertAlmostEqual(ca.calc_precision("macro"),0.305555556, places=4)

        #check recall
        self.assertAlmostEqual(ca.calc_recall("macro"), 0.305555556, places=4)

        #check f1
        self.assertAlmostEqual(ca.calc_f1_score("macro"), 0.305555556, places=4)

    

if __name__ == '__main__':
    # Grabs the location of the unit test file and sets cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    unittest.main()