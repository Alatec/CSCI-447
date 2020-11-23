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
        # self.train_data = train

        if prediction_type == 'regression':
            bin_edges = np.histogram(self.unknown_col, bins=bins)[1]
            self.unknown_col = pd.Series(np.digitize(self.unknown_col, bin_edges), dtype=str)

        self.unique_classes = self.unknown_col.unique()
    def train(self):
        
        #Contains values counts for each combination of attributes
        self.attrDict = {}
        #Contrains the delta (relative probability) matrix
        self.probMatrix = {}

        #Populate the attrDict
        #Iterate through each column
        for i, col in enumerate(self.train_data.columns):
            self.attrDict[col] = {}
            values = list(self.train_data[col].unique())
            values.append("unseen")
            #Iterate through each unique val
            for j, val in enumerate(values):
                self.attrDict[col][val] = {
                    "total_VDM": 0
                }
                #Iterate through each class
                for k, clss in enumerate(self.unique_classes):
                    self.attrDict[col][val][clss] = len(self.train_data[(self.train_data[col].astype(str)==str(val)) & (self.unknown_col==clss)]) + 1
                    self.attrDict[col][val]["total_VDM"] += self.attrDict[col][val][clss]
                
        
        for i, col in enumerate(self.train_data.columns):
            # print(col)
            self.probMatrix[col]  = {}
            unique_vals = list(self.train_data[col].unique())
            unique_vals.append("unseen")
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
                    x_val = "unseen"
                    y_val = "unseen"

                    if x_row[k] in self.probMatrix[col]: x_val = x_row[k]
                    if y_row[k] in self.probMatrix[col]: y_val = y_row[k]

                    self.distances[i, j] += self.probMatrix[col][x_val][y_val]
                    



if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("computerHardware")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)[["venderName", "modelName", "myct", "mmin", "mmax", "cach", "chmin", "chmax", "prp", "erp"]]

    test = data.sample(frac=0.2)
    train = data.drop(test.index)

    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)


    VDM = ValueDifferenceMetric(data, unknown_col=dataRetriever.getDataClass(), prediction_type=dataRetriever.getPredictionType())

    start = time.time()
    VDM.train()
    print(f"Training took: {time.time() - start} seconds")

    start = time.time()
    VDM.calc_distance_matrix(data["mmin"], data["mmin"])
    print(f"Matrix took: {time.time() - start} seconds")

    # print(KNN.get_neighbors([5,10]))
    # print(KNN.test(augmented=True))