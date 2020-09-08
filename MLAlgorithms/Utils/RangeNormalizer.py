import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from MLAlgorithms.Errors.UtilError import *


class RangeNormalizer:
    
    def __init__(self, train_data):
        """Maps dataset from [0,1]

        Args:
            train_data (DataFrame, numpy array): The dataset to be normalized
        """
        self.train_data = train_data
        self.parameters = {}
        
    def train(self):
        self.parameters["min"] = self.train_data.min()
        self.parameters["max"] = self.train_data.max()
        self.parameters["range"] = self.parameters["max"] - self.parameters["min"]

    def fit(self, data):
        """Fit unseen data according to the trained model

        Args:
            data (DataFrame): data to fit

        Raises:
            UntrainedUtilityError: Raised if fit is called before train
            TrainTestColumnMismatch: Raised if the number of columns in the test set is different than in the train set

        Returns:
            DataFrame: Range normalized data
        """


        #Error checking
        if "min" not in self.parameters: raise UntrainedUtilityError

        length_check = 0
        if len(data.shape) == 1:
            length_check = 1
        else:
            length_check = data.shape[1]
        #if len(self.parameters["min"]) != length_check: raise TrainTestColumnMismatch
        
        #Calculate Z-Score of test data in terms of train data parameters
        return (data-self.parameters["min"])/self.parameters["range"]

    def train_fit(self):
        self.train()
        return self.fit(self.train_data)