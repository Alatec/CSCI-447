from numba.experimental import jitclass
from numba import int32, float32, float64 
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


dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
data = dataRetriever.getDataSet()
data = data.dropna()
data = data.reset_index(drop=True)


spec = [
    ('test', float64[:,:]),
    ('train', float64[:,:])
]
@jitclass(spec)
class DistanceCalculator(object):
    def __init__(self, test, train):
        self.train = train
        self.test = test

    def calculate_distances(self):
        distances = np.zeros((self.test.shape[0], self.train.shape[0]), dtype=np.float64)

        for i in range(self.test.shape[0]):
            for j in range(self.train.shape[0]):
             distances[i,j] = ((self.test[i]-self.train[j])**2).sum()

        return distances


class KNearestNeighbor:
    def __init__(self,test, train, k, metric='euclid'):
        self.test = test
        self.train = train
        self.k = k
        self.distance_calc = DistanceCalculator(test.to_numpy(), train.to_numpy())

    def get_neighbors(self, test_index):
        neighbors = self.distance_calc.calculate_distances()
        
        return neighbors[test_index].argsort()[:self.k]

if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("breastCancer")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)

    KNN = KNearestNeighbor(data,data,5)
    print(KNN.get_neighbors(5))
        