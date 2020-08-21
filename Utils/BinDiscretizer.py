import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinDiscretizer:
    
    def __init__(self, bins, train_data):
        self.bins = bins
        self.train_data = train_data

    def train(self, bins='auto'):
        """Will generate fixed width bins for an input dataset

        Args:
            bins ([int, str], optional): if int, will generate that number of bins. If str, will use algorithm from numpy.histogram_bin_edges Defaults to 'auto'.
        """

        