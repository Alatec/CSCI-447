import pandas as pd
import numpy as np
import time 

from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances

class ValueDifferenceMetric:
    def __init__(self, train, unknown_col='class', bins=8, prediction_type='classification'):
        """
        docstring
        """

        self.unknown_col = train[unknown_col].reset_index(drop=True)
        self.train_data = train.drop(unknown_col, axis=1).reset_index(drop=True)

        if prediction_type == 'regression':
            bin_edges = np.histogram(self.unknown_col, bins=bins)[1]
            self.unknown_col = pd.Series(np.digitize(self.unknown_col, bin_edges), dtype=str)

        self.unique_classes = self.unknown_col.unique()
    def train(self):
        """
        docstring
        """
        self.attrDict = {}
        self.probMatrix = {}
        for i, col in enumerate(self.train_data.columns):
            self.attrDict[col] = {}
            for j, val in enumerate(self.train_data[col].unique()):
                self.attrDict[col][val] = {
                    "total_VDM": 0
                }
                for k, clss in enumerate(self.unique_classes):
                    self.attrDict[col][val][clss] = len(self.train_data[(self.train_data[col]==val) & (self.unknown_col==clss)])
                    self.attrDict[col][val]["total_VDM"] += self.attrDict[col][val][clss]
        
        for i, col in enumerate(self.train_data.columns):
            # print(col)
            self.probMatrix[col]  = {}
            unique_vals = self.train_data[col].unique()
            for val1 in unique_vals:
                self.probMatrix[col][val1] = {}
                for val2 in unique_vals:
                    # print(self.probMatrix)
                    self.probMatrix[col][val1][val2] = 0
                    for clss in self.unique_classes:
                        self.probMatrix[col][val1][val2] += abs(self.attrDict[col][val1][clss]/self.attrDict[col][val1]["total_VDM"]
                                             - self.attrDict[col][val2][clss]/self.attrDict[col][val2]["total_VDM"])

    def calc_distance_matrix(self, x_points, y_points):
        self.distances = np.zeros((len(x_points), len(y_points)))
        test_cols = ""
        if type(x_points.iloc[0]) != pd.Series:
            test_cols = list(self.probMatrix.keys())
        else:
            test_cols = x_points.columns


        for i, x_row in enumerate(x_points.to_numpy()):
            for j, y_row in enumerate(y_points.to_numpy()):
                for k, col in enumerate(test_cols):
                    # print(col, type(i) , type(j))
                    # print(type(x_points.iloc[i]))
                    
                    self.distances[i,j] += self.probMatrix[col][x_row[k]][y_row[k]]
                    



if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("abalone")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)[["sex","rings"]]

    test = data.sample(frac=0.2)
    train = data.drop(test.index)

    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)


    VDM = ValueDifferenceMetric(data, unknown_col='rings', prediction_type='regresssion')

    start = time.time()
    VDM.train()
    print(f"Training took: {time.time() - start} seconds")

    start = time.time()
    VDM.calc_distance_matrix(data["sex"], data["sex"])
    print(f"Matrix took: {time.time() - start} seconds")

    # print(KNN.get_neighbors([5,10]))
    # print(KNN.test(augmented=True))