import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from MLAlgorithms.Errors.UtilError import *


class StandardNormalizer:
    
    def __init__(self, train_data):
        """Converts the dataset to its Z-Score

        Args:
            train_data (DataFrame, numpy array): The dataset to be normalized
        """
        self.train_data = train_data
        self.parameters = {}
        
    def train(self):
        self.parameters["mean"] = self.train_data.mean()
        self.parameters["sd"] = self.train_data.std()

    def fit(self, data):
        """Fit unseen data according to the trained model

        Args:
            data (DataFrame): data to fit

        Raises:
            UntrainedUtilityError: Raised if fit is called before train
            TrainTestColumnMismatch: Raised if the number of columns in the test set is different than in the train set

        Returns:
            DataFrame: Z-Score normalized data
        """


        #Error checking
        if "mean" not in self.parameters: raise UntrainedUtilityError

        length_check = 0
        if len(data.shape) == 1:
            length_check = 1
        else:
            length_check = data.shape[1]
        
        if len(self.parameters["mean"]) != length_check: raise TrainTestColumnMismatch
        
        #Calculate Z-Score of test data in terms of train data parameters
        return (data-self.parameters["mean"])/self.parameters["sd"]

    def train_fit(self):
        self.train()
        return self.fit(self.train_data)