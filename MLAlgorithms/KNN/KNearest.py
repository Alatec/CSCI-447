import numpy as np
import pandas as pd

import matplotlib
print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances

class KNearestNeighbor:
    def __init__(self, test, train, k, unknown_col='class', metric='euclid', classification=True):
        self.test_data = test
        self.train_data = train
        self.k = k
        self.metric = metric

        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop([unknown_col], axis=1)


        if metric == 'euclid': 
            self.neighbors = calculate_euclid_distances(self.test_data.to_numpy(dtype=np.float64), self.train_data.to_numpy(dtype=np.float64))
        else:
            raise NotImplementedError

    def get_neighbors(self, test_index):        
        return self.neighbors[test_index].argsort()[:self.k]
    
    #
    def get_most_common_class_apply(self, neighbors_list):
        k_neighbors = neighbors_list.argsort()[:self.k]
        unknowns = self.unknown_col.loc[k_neighbors]
        return unknowns.value_counts().idxmax()


    def test(self):
        return np.apply_along_axis(self.get_most_common_class_apply,0,self.neighbors)

if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("breastCancer")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)

    KNN = KNearestNeighbor(data.drop('class', axis=1),data,5)
    print(KNN.get_neighbors(5))
    print(KNN.test())
        