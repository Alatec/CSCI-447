import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinDiscretizer:
    
    def __init__(self,  train_data, bins='auto', multi=False):

        """Intended to separate continuous data into discrete bins

        Args:
            train_data (numpy array): the data to be discritized
            bins ([int, str]): Method used to bin. See numpy.histogram_bin_edges
        """

        self.bins = bins
        self.train_data = train_data
        self.trained = False
        self.multi = multi
        self.columns = None
        # self.discritizers = {}

    def train(self, bins=None):
        """Will generate fixed width bins for an input dataset

        Args:
            bins ([int, str], optional): if int, will generate that number of bins. If str, will use algorithm from numpy.histogram_bin_edges Defaults to 'auto'.
        """
        if self.multi:
            print("This method is intend for 1 column at a time")
        if bins is not None:
            self.bins = bins
        self.bin_edges = np.histogram_bin_edges(self.train_data, bins=self.bins)
        self.trained = True

    def fit(self, data):
        """Returns the bins that each data point would be in based off of the training set

        Args:
            data (numpy array): data to be fitted
        """
        if self.multi:
            print("This method is intend for 1 column at a time")
        if self.trained:
            return np.digitize(data, bins=self.bin_edges)
        else:
            print("This has not been trained")
            print("Probably should make this an actual error so that the code stops")

    def train_fit(self):
        self.train()
        return self.fit(self.train_data)

    def train_multi(self, columns=None, bins=None):

        self.discritizers = {}

        if not self.multi:
            print("This method is ended for multiple columns and might behave unexpectedly")

        if bins is not None:
            self.bins = bins

        if columns is None:
            columns = self.train_data.columns
        
        self.columns = columns

        if type(self.bins) is not list:
            self.bins = [self.bins] * len(columns)
        else:
            assert len(self.bins) == len(columns)

        for i, col in enumerate(columns):
            self.discritizers[col] = {}
            self.discritizers[col]["bin_edges"] = np.histogram_bin_edges(self.train_data[col], bins=self.bins[i])

        self.trained = True

    def fit_multi(self, test_data):
        """Returns the bins that each data point would be in based off of the training set

        Args:
            data (numpy array): data to be fitted
        """
        if not self.multi:
            print("This method is ended for multiple columns and might behave unexpectedly")
        if self.trained:
            return test_data[self.columns].apply(lambda x: self._fit_apply(x))
        else:
            print("This has not been trained")
            print("Probably should make this an actual error so that the code stops")

    def _fit_apply(self, data):
        #This is only it's own function in case we need more control
        return np.digitize(data, bins=self.discritizers[data.name]["bin_edges"])