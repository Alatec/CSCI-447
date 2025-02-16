import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the Root directory to the import path. May cause linters to complain about finding classes
# import sys
# sys.path.append('../..')

from MLAlgorithms.Errors.UtilError import *

class BinDiscretizer:
    
    def __init__(self,  train_data, bins=8, multi=False):

        """Intended to separate continuous data into discrete bins

        Args:
            train_data (numpy array): the data to be discritized
            bins (int): Number of bins to use
        """

        self.bins = bins
        self.train_data = train_data
        self.trained = False
        self.multi = multi
        self.columns = None

        #Type checking
        if self.bins is None:
            print("bins cannot be None")
            raise TypeError
        # self.discritizers = {}

    def train(self, bins=None):
        """Will generate fixed width bins for an input dataset

        Args:
            bins (int, optional): Number of bins to generate
        """
        if self.multi:
            raise SingleMultiError(self.multi)
        if bins is not None:
            self.bins = bins
        self.bin_edges = self._calculate_bins(self.train_data, bins=self.bins)
        self.trained = True

    def fit(self, data):
        """Returns the bins that each data point would be in based off of the training set

        Args:
            data (numpy array): data to be fitted
        """
        if self.multi:
            raise SingleMultiError(self.multi)
        if self.trained:
            return self._digitize_from_bins(data, self.bin_edges)
        else:
            # This as not been trained
            raise UntrainedUtilityError("BinDiscritzer")
            # print("This has not been trained")
            # print("Probably should make this an actual error so that the code stops")

    def train_fit(self):
        self.train()
        return self.fit(self.train_data)

    def train_multi(self, columns=None, bins=None):

        self.discritizers = {}

        if not self.multi:
            raise SingleMultiError(self.multi)

        if bins is not None:
            self.bins = bins

        if columns is None:
            columns = self.train_data.columns
        
        self.columns = columns

        if type(self.bins) is not list:
            self.bins = [self.bins] * len(columns)
        else:
            if len(self.bins) != len(columns):
                print("bins and columns must be the same size!")
                raise ValueError

        for i, col in enumerate(columns):
            self.discritizers[col] = {}
            self.discritizers[col]["bin_edges"] = self._calculate_bins(self.train_data[col], bins=self.bins[i])

        self.trained = True

    def fit_multi(self, test_data):
        """Returns the bins that each data point would be in based off of the training set

        Args:
            data (DataFrame): data to be fitted
        """
        if not self.multi:
            raise SingleMultiError(self.multi)
        if self.trained:
            return test_data[self.columns].apply(lambda x: self._fit_apply(x))
        else:
            # This as not been trained
            raise UntrainedUtilityError("Multi BinDiscritzer")
            
    def _fit_apply(self, data):
        #This is only it's own function in case we need more control
        return self._digitize_from_bins(data, self.discritizers[data.name]["bin_edges"])

    def _calculate_bins(self, data, bins):
        data_range = [data.min(), data.max()]
        edges = np.linspace(*data_range, num=(bins+1))

        return edges[1:]

    def _digitize_from_bins(self, data, bin_edges):
        sorted_args = np.argsort(data, kind='stable')
        sorted_data = np.sort(data, kind='stable')
        digitized_data = np.zeros_like(data) 

        left_edge = data.min()
        last_edge = len(bin_edges) - 1
        for i, edge in enumerate(bin_edges):
            indices = None
            if i == last_edge:
                indices = np.where(sorted_data>left_edge)
                indices = np.asarray(indices, dtype=np.int64).flatten()
                if len(indices):
                    digitized_data[sorted_args[indices[0]:]] = i
            else:
                indices = np.where((sorted_data>=left_edge)&(sorted_data<edge))
                indices = np.asarray(indices, dtype=np.int64).flatten()
                # print(indices)
                if len(indices):
                    digitized_data[sorted_args[indices[0]:]] = i

            left_edge = edge
            
        return digitized_data + 1
            
            

