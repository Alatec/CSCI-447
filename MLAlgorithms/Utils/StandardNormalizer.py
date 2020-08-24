import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from Errors.UtilError import *


class StandardNormalizer:
    
    def __init__(self, train_data, multi=True):
        """Converts the dataset to its Z-Score

        Args:
            train_data (DataFrame, numpy array): The dataset to be normalized
            multi (bool, optional): Tells the normalizer whether it is for multiple columns. Defaults to True.
        """
        self.train_data = train_data
        self.multi = multi
        